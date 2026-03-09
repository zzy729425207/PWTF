import time

import torch
from typing import List
import math
import torch.nn as nn
import torch.nn.functional as F
from .extractor import ContextNet, DepthAnythingFeature, ResNet18Deconv, resconv

from .update import BasicUpdateBlock, VisionTransformer
from .submodule import *


class LocalCorr:
    def __init__(self, fmap1, fmap2, radius=4, lr=16):
        self.radius = radius
        self.lr = lr
        corr = LocalCorr.corr(fmap1, fmap2, lr)
        b, h, w, _, _ = corr.shape
        self.corr = corr.reshape(b * h * w, 1, 2 * lr + 1, 2 * lr + 1)

    def __call__(self, flow):
        r = self.radius
        coords = flow + self.lr
        coords = coords.permute(0, 2, 3, 1)
        b, h, w, _ = coords.shape
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        centroid_lvl = coords.reshape(b * h * w, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl
        corr = bilinear_sampler(self.corr, coords_lvl)
        corr = corr.view(b, h, w, -1)
        return corr.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2, lr):
        b, c, h, w = fmap1.shape
        fmap2 = F.pad(fmap2, (lr, lr, 0, 0))
        fmap2 = fmap2.unfold(dimension=3, size=2 * lr + 1, step=1)  # [B, C, H, W, 2*r+1]
        corr = fmap1.new_zeros([b, h, w, 2 * lr + 1, 2 * lr + 1])
        fmap1 = fmap1.unsqueeze(4)
        for i in range(-lr, lr + 1):
            if i < 0:

                corr[:, -i:, :, i + lr, :] = (fmap1[:, :, -i:, :, :] * fmap2[:, :, :i, :, :]).sum(dim=1)
            elif i == 0:
                corr[:, :, :, i + lr, :] = (fmap1 * fmap2).sum(dim=1)
            else:

                corr[:, :-i, :, i + lr, :] = (fmap1[:, :, :-i, :, :] * fmap2[:, :, i:, :, :]).sum(dim=1)
        return corr / torch.sqrt(torch.tensor(c).float())


def MLP_no_ReLU(channels: List[int], do_bn: bool = False) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
    return nn.Sequential(*layers)


def TrackFine(fmap1, fmap2, lr):
    """
    使用unfold和爱因斯坦求和的完全向量化实现
    """
    b, c, h, w = fmap1.shape

    # 在fmap2周围填充
    pad_h = lr
    fmap2_padded = F.pad(fmap2, (lr, lr, pad_h, pad_h))

    # 将fmap2展开为局部块
    # 展开尺寸：垂直方向 [2*lr+1]，水平方向 [2*lr+1]
    fmap2_unfolded = fmap2_padded.unfold(2, 2 * lr + 1, 1).unfold(3, 2 * lr + 1, 1)  # [B, C, H, W, 2*lr+1, 2*lr+1]

    # 调整fmap1维度用于广播
    fmap1_expanded = fmap1.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W, 1, 1]

    # 计算点积（爱因斯坦求和优化）
    corr = torch.einsum('bchwij,bchwij->bhwij',
                        fmap1_expanded.expand_as(fmap2_unfolded),
                        fmap2_unfolded)

    return corr / torch.sqrt(torch.tensor(c).float())


class NormalEncoder(nn.Module):  # 192, 64, [128, 64, 64], dropout=False
    """ Encoding of geometric properties using MLP """

    def __init__(self, normal_dim: int, feature_dim: int, layers: List[int], dropout: bool = False,
                 p: float = 0.1) -> None:
        super().__init__()
        self.encoder = MLP_no_ReLU([normal_dim] + layers + [feature_dim])
        # [192] + [128, 64, 64] + [64]  -->  [192, 128, 64, 64, 64]
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, kpts):
        if self.use_dropout:
            return self.dropout(self.encoder(kpts))
        return self.encoder(kpts)


def compute_normal_map_torch(
        depth_map: torch.Tensor,
        scale: float = 1.0,
        use_sobel: bool = True,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    通过深度图计算法向量（适配输入维度 (B,1,320,640)，强制校验尺寸）

    参数：
        depth_map (torch.Tensor): 深度图，仅支持 (B,1,320,640) 维度
        scale (float): 深度值的比例因子，调整梯度幅度
        use_sobel (bool): 是否使用Sobel算子计算梯度（抗噪声，默认开启）
        eps (float): 防止除零的极小值

    返回：
        torch.Tensor: 法向量图，形状为 (B, 320, 640, 3)
    """
    # -------------------------- 1. 严格校验输入维度和尺寸 --------------------------
    if depth_map.ndim != 4:
        raise ValueError(f"输入深度图维度必须为 (B,1,320,640)，当前维度为 {depth_map.ndim}")
    B, C, H, W = depth_map.shape

    device = depth_map.device
    dtype = depth_map.dtype

    # Sobel算子（3x3，带权重，抗噪声，适配320x640尺寸）
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)

    # 直接卷积（输入已带通道维 (B,1,320,640)，无需额外扩展）
    # padding=1 保证输出仍为 320x640，边缘处理更平滑
    dzdx = F.conv2d(depth_map, sobel_x, padding=1).squeeze(1)  # (B,320,640)
    dzdy = F.conv2d(depth_map, sobel_y, padding=1).squeeze(1)  # (B,320,640)

    # 应用比例因子
    dzdx = dzdx * scale
    dzdy = dzdy * scale

    # -------------------------- 3. 构造并归一化法向量 --------------------------
    # 初始化法向量：(B, 320, 640, 3)
    normal_map = torch.zeros((B, H, W, 3), dtype=dtype, device=device)
    normal_map[..., 0] = -dzdx  # x分量（梯度取负）
    normal_map[..., 1] = -dzdy  # y分量（梯度取负）
    normal_map[..., 2] = 1.0  # z分量（垂直图像平面）

    norm = torch.linalg.norm(normal_map, dim=-1, keepdim=True)
    normal_map = normal_map / torch.maximum(norm, torch.tensor(eps, dtype=dtype, device=device))

    normal_map = normal_map.permute(0, 3, 1, 2)

    return normal_map


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # self.sparse = sparse
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.bn3 = norm_layer(planes)
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                self.bn3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class ResNetFPN(nn.Module):
    """
    ResNet18, output resolution is 1/8.
    Each block has 2 layers.
    """

    def __init__(self, input_dim=3, output_dim=256, ratio=1.0, norm_layer=nn.BatchNorm2d, init_weight=False):
        super().__init__()
        # Config
        block = BasicBlock
        block_dims = [64, 128, 256]
        initial_dim = 64
        self.init_weight = init_weight
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
        # Networks
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        n_block = [2, 2, 2]

        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2])  # 1/8
        self.final_conv = conv1x1(block_dims[2], output_dim)
        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.init_weight:
            from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
            pretrained_dict = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if self.input_dim == 6:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        pretrained_dict[k] = torch.cat((v, v), dim=1)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
        # Output
        output = self.final_conv(x)
        return output


class Model(nn.Module):
    def __init__(self,
                 downsample_factor=8,
                 feature_channels=256,
                 hidden_dim=128,
                 context_dim=128,
                 corr_radius=4,
                 mixed_precision=False,
                 **kwargs,
                 ):
        super(Model, self).__init__()

        self.downsample_factor = downsample_factor
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_radius = corr_radius
        self.mixed_precision = mixed_precision

        self.encoder = DepthAnythingFeature(model_name="vits", pretrained=True, lvl=-3)
        self.factor = 112

        self.pretrain_dim = self.encoder.output_dim

        self.fnet = ResNet18Deconv(3, self.pretrain_dim)

        self.fmap_conv = nn.Conv2d(self.pretrain_dim // 2 * 3 + 3, 48, kernel_size=1, stride=1, padding=0, bias=True)

        self.downsample_fmap = resconv(48, 64, k=3, s=2)

        self.hidden_conv = nn.Conv2d(48 * 2 + 3 * 2, 48, kernel_size=1, stride=1, padding=0, bias=True)

        self.downsample_net = resconv(48,64,k=3,s=2)

        self.hidden_conv_fine = nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1, padding=0, bias=True)

        self.warp_linear = nn.Conv2d(3 * 48 + 2 * 3 + 2, 64, 1, 1, 0, bias=True)

        self.warp_linear_fine = nn.Conv2d(2 * 64, 64, 1, 1, 0, bias=True)

        self.refine_net = VisionTransformer("vitt", 64, patch_size=8)

        self.Track_net = BasicUpdateBlock(25)

        self.refine_transform = nn.Conv2d(64, 48, 1, 1, 0, bias=True)

        # self.cnet = ContextNet(output_dim=hidden_dim + context_dim, norm_fn='batch')
        # self.cnet = ResNetFPN(input_dim=6, output_dim=2 * 128, norm_layer=nn.BatchNorm2d, init_weight=True)
        self.init_conv = conv3x3(2 * 128, 2 * 128)

        # 1D attention
        corr_channels = (2 * corr_radius + 1) * 2 * 8 + (2 * corr_radius + 1) ** 2

        self.conv_down = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1)
        self.CostRegNet = CostRegNet(8, 8)
        self.mask_init = nn.Sequential(
            nn.Conv2d(feature_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, downsample_factor ** 2 * 9, 1, padding=0))

        # Update block
        self.update_block = BasicUpdateBlock(corr_channels, hidden_dim)
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(48, 2 * 48, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * 48, 6, 1, padding=0, bias=True)
        )

        self.flow_head_fine= nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(64, 2 * 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * 64, 6, 1, padding=0, bias=True)
        )

        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(48, 2 * 48, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * 48, 4 * 9, 1, padding=0, bias=True)
        )
        self.upsample_weight_fine = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(64, 2 * 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * 64, 16 * 9, 1, padding=0, bias=True)
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, downsample=None):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        n, c, h, w = img.shape
        downsample_factor = self.downsample_factor if downsample is None else downsample
        coords0 = coords_grid(n, h // downsample_factor, w // downsample_factor).to(img.device)
        coords1 = coords_grid(n, h // downsample_factor, w // downsample_factor).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 2 * H, 2 * W), up_info.reshape(N, C, 2 * H, 2 * W)

    def upsample_data_fine(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 4 * H, 4 * W), up_info.reshape(N, C, 4 * H, 4 * W)

    def forward(self, image1, image2, iters=4, flow_gt=None, test_mode=False, Track_Fine=True):
        """ Estimate optical flow between pair of frames """
        iters_P = 4
        iters_T = 4
        Track_Fine = True

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        padder = Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)

        N, _, H, W = image1.shape

        flow_predictions = []
        info_predictions = []

        # run the feature network
        fmap1_pretrain, _, depth1 = self.encoder(image1)
        fmap2_pretrain, _, depth2 = self.encoder(image2)

        depth1 = (depth1 - depth1.min()) / (depth1.max() - depth1.min()) * 255.0
        depth2 = (depth2 - depth2.min()) / (depth2.max() - depth2.min()) * 255.0

        normal_map1 = compute_normal_map_torch(depth1, 1.0)
        normal_map2 = compute_normal_map_torch(depth2, 1.0)

        fmap1_img = self.fnet(image1)[0]
        fmap2_img = self.fnet(image2)[0]

        # 99-64 torch.Size([1, 64, 168, 336])
        fmap1_2x = self.fmap_conv(torch.cat([fmap1_pretrain, fmap1_img, normal_map1], dim=1))
        fmap2_2x = self.fmap_conv(torch.cat([fmap2_pretrain, fmap2_img, normal_map2], dim=1))

        # 增强表面法向量在预测中的作用
        net = self.hidden_conv(torch.cat([fmap1_2x, normal_map1, fmap2_2x, normal_map2], dim=1))  # 128 + 6 - 48

        flow_2x = torch.zeros(N, 2, H // 2, W // 2).to(image1.device)  # 初始化 0

        for itr in range(iters_P):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H // 2, W // 2, device=image1.device) + flow_2x).detach()
            # 生成采样位置

            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            # 使用warp操作对fmap2_2x采样
            warp_normal2 = bilinear_sampler(normal_map2, coords2.permute(0, 2, 3, 1))

            refine_inp = self.warp_linear(
                torch.cat([fmap1_2x, normal_map1, warp_2x, warp_normal2, net, flow_2x], dim=1))
            # torch.Size([1, 64, 168, 336])

            refine_outs = self.refine_net(refine_inp)  # torch.Size([1, 16, 168, 336])
            # 操作特征  本质是使用Transformer进行长程搜索的光流初始化  没有指定的意义

            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))
            # 线性处理

            if test_mode == True and itr != iters - 1:
                continue
            else:
                flow_update = self.flow_head(net)
                # net中预测update

                weight_update = .25 * self.upsample_weight(net)
                # 生成上采样权重

                flow_2x = flow_2x + flow_update[:, :2]
                info_2x = flow_update[:, 2:]

                # upsample predictions
                flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

            if itr == iters_P - 1 and Track_Fine == True:  # 循环的最后一次
                flow_2x = flow_2x.detach()
                flow_4x = F.interpolate(0.5 * flow_2x, scale_factor=0.5, mode='nearest')
                net = net.detach()
                coords2 = (coords_grid(N, H // 2, W // 2, device=image1.device) + flow_2x).detach()
                warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))  # fmap1_2x

                # 这个特征可以从上面读取
                fmap1_4x = self.downsample_fmap(fmap1_2x)  # 64
                fmap2_4x = self.downsample_fmap(warp_2x)  # 64

                local_corr = LocalCorr(fmap1_4x, fmap2_4x, radius=2, lr=5)

                net_down = self.downsample_net(net)
                net_fine = net_down + self.hidden_conv_fine(torch.cat([fmap1_4x, fmap2_4x], dim=1))  # 64

                for detial_iter in range(iters_T):
                    flow_4x = flow_4x.detach()
                    corr = local_corr(flow_4x)  # 1 25 84 168  里面的还没通道还没调整好，名字还没调整
                    #  64 * 3 +2 + 25
                    refine_inp = self.warp_linear_fine(torch.cat([fmap1_4x, fmap2_4x], dim=1))  # 128-64
                    net_fine = self.Track_net(net_fine, refine_inp, corr, flow_4x)  # 这里使用SEARAFT的Conxet

                    if test_mode == True and detial_iter != iters_T - 1:
                        continue
                    else:
                        flow_update_fine = self.flow_head_fine(net_fine)
                        # net中预测update
    
                        weight_update_fine = .25 * self.upsample_weight_fine(net_fine)
                        # 生成上采样权重
    
                        flow_4x = flow_4x + flow_update_fine[:, :2]
                        info_4x = flow_update_fine[:, 2:]
    
                        # upsample predictions
                        flow_up_fine, info_up_fine = self.upsample_data_fine(flow_4x, info_4x, weight_update_fine)
                    flow_predictions.append(flow_up_fine)
                    info_predictions.append(info_up_fine)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):
                var_max = 10
                var_min = 0

                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2,
                                                                                         dim=2)
                nf_predictions.append(nf_loss)
            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions,
                    'nf': nf_predictions}
        else:
            return {'final': flow_predictions[-1]}


def build_model(args):
    return Model(downsample_factor=args.downsample_factor,
                 feature_channels=args.feature_channels,
                 corr_radius=args.corr_radius,
                 hidden_dim=args.hidden_dim,
                 context_dim=args.context_dim,
                 mixed_precision=args.mixed_precision,
                 )
