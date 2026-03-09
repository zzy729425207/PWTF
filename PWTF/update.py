import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from .patch_embed import PatchEmbed
from .DepthAnythingV2.depth_anything_v2.dpt import DPTHead

MODEL_CONFIGS = {
    'vitl': {'encoder': 'vit_large_patch16_224', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vit_base_patch16_224', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vit_small_patch16_224', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitt': {'encoder': 'vit_tiny_patch16_224', 'features': 32, 'out_channels': [24, 48, 96, 192]}
}

class VisionTransformer(nn.Module):
    def __init__(self, model_name, input_dim, patch_size=16):
        super(VisionTransformer, self).__init__()
        model = timm.create_model(
            MODEL_CONFIGS[model_name]['encoder'],
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        self.intermediate_layer_idx = {
            'vitt': [2, 5, 8, 11],
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }
        self.idx = self.intermediate_layer_idx[model_name]
        self.blks = model.blocks
        self.embed_dim = model.embed_dim
        self.input_dim = input_dim
        self.img_size = (224, 224)
        self.patch_size = patch_size
        self.output_dim = MODEL_CONFIGS[model_name]['features']
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, self.embed_dim))
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=input_dim, embed_dim=self.embed_dim)
        self.dpt_head = DPTHead(self.embed_dim, MODEL_CONFIGS[model_name]['features'], out_channels=MODEL_CONFIGS[model_name]['out_channels'])

    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sy, sx),
            mode="bicubic",
            antialias=False
        )
        assert int(w0) == pos_embed.shape[-1]
        assert int(h0) == pos_embed.shape[-2]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)

    def forward(self, x):
        B, nc, h, w = x.shape
        x = self.patch_embed(x)  # 切割成若干个互相不重叠的正方向小Patch  1 21*42 C = 384
        x = x + self.interpolate_pos_encoding(x, h, w)  # 添加相对位置编码
        outputs = []
        for i in range(len(self.blks)):
            x = self.blks[i](x)
            if i in self.idx:
                outputs.append([x])  # 提取中间特征 4 次， 每次提取到的特征 torch.Size([1, 882, 384])
        # torch.Size([1, 882, 384]) torch.Size([1, 882, 384]) torch.Size([1, 882, 384]) torch.Size([1, 882, 384])

        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        out, depth, path_1, path_2, path_3, path_4 = self.dpt_head.forward(outputs, patch_h, patch_w)

        # torch.Size([1, 32, 294, 588]) out
        # torch.Size([1, 64, 168, 336]) path_1
        # torch.Size([1, 64, 84, 168]) path_2
        # torch.Size([1, 64, 42, 84]) path_3
        # torch.Size([1, 64, 21, 42]) path_4

        # DPTHead输出密集预测结果
        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256,
                 ):
        super(FlowHead, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)

        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))

        return out


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128,
                 kernel_size=5,
                 ):
        padding = (kernel_size - 1) // 2

        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_channel, dim=128):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_channel
        self.convc1 = nn.Conv2d(cor_planes, dim*2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim*2, dim+dim//2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim//2, 3, padding=1)
        self.conv = nn.Conv2d(dim*2, dim-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, output_dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * output_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * output_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.final(input + x)
        return x

class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_channel, hdim=64, cdim=64):
        #net: hdim, inp: cdim
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_channel, cdim)
        self.refine = []
        for i in range(2):
            self.refine.append(ConvNextBlock(2*cdim+hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)  # 2+25 -> 48
        inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net
