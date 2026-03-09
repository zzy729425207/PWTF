import torch
import torch.nn as nn
import torch.nn.functional as F
from .head import DPTHead
from .DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

DEPTH_ANYTHING_CONFIGS = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}


class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k // 2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)


class ResNet18Deconv(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18Deconv, self).__init__()
        self.feature_dims = [64, 128, 256, 512]
        self.ds1 = resconv(inp, 64, k=7, s=2)
        self.conv1 = resconv(64, 64, k=3, s=1)
        self.conv2 = resconv(64, 128, k=3, s=2)
        self.conv3 = resconv(128, 256, k=3, s=2)
        self.conv4 = resconv(256, 512, k=3, s=2)
        self.up_4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_3 = resconv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_2 = resconv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_1 = resconv(64, oup, k=3, s=1)

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))
        return [out_1, out_2, out_3, out_4]


class DepthAnythingFeature(nn.Module):
    def __init__(self, model_name='vits', pretrained=True, lvl=-3):
        super().__init__()
        self.dpt_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        self.model_name = model_name
        depth_anything = DepthAnythingV2(**DEPTH_ANYTHING_CONFIGS[model_name])
        if pretrained:
            depth_anything.load_state_dict(torch.load(f'/home/johndoe/data-1/jogndoe/Optical/HCVFlow/depth_anything_v2_{model_name}.pth', map_location='cpu'))

        self.encoder = self.freeze_(depth_anything.pretrained)
        self.monodecocer = self.freeze_(depth_anything.depth_head)
        self.embed_dim = depth_anything.pretrained.embed_dim
        self.output_dim = self.dpt_configs[model_name]['features']
        self.out_channels = self.dpt_configs[model_name]['out_channels']
        del depth_anything
        self.dpt_head = DPTHead(self.embed_dim, features=self.output_dim, out_channels=self.out_channels, lvl=lvl)

    def freeze_(self, model):
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        return model

    def forward(self, x):
        """
        @x: (B,C,H,W)
        """
        h, w = x.shape[-2], x.shape[-1]
        features = self.encoder.get_intermediate_layers(x, self.intermediate_layer_idx[self.model_name],
                                                        return_class_token=True)
        patch_size = self.encoder.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        outs, depth, _, _, _, _ = self.monodecocer(features, patch_h, patch_w)
        outs_ = self.dpt_head.forward(features, patch_h, patch_w)
        depth = F.relu(depth)
        depth = F.interpolate(depth, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
        final = F.interpolate(outs, size=(h // 2, w // 2), mode='bilinear', align_corners=True)  # freeze fea
        outs_ = F.interpolate(outs_[0], size=(h//2, w//2), mode='bilinear', align_corners=True)  # activate fea
        return final, outs_, depth


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, dilation=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1 or in_planes != planes:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1 or in_planes != planes:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class FeatureNet(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0,
                 **kwargs,
                 ):
        super(FeatureNet, self).__init__()
        self.norm_fn = norm_fn

        feature_dims = [64, 96, 128, 160]

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=feature_dims[0])

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(feature_dims[0])

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(feature_dims[0])

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1)
        self.layer2 = self._make_layer(feature_dims[1], stride=2)  # 1/4

        self.layer3 = self._make_layer(feature_dims[2], stride=2, dilation=1)

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, kernel_size=1)

        # self.conv_down = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1, dilation=dilation)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4

        x = self.layer3(x)  # 1/8

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class ContextNet(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0,
                 **kwargs,
                 ):
        super(ContextNet, self).__init__()
        self.norm_fn = norm_fn

        feature_dims = [64, 96, 128, 160]

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=feature_dims[0])

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(feature_dims[0])

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(feature_dims[0])

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1)
        self.layer2 = self._make_layer(feature_dims[1], stride=2)  # 1/4

        self.layer3 = self._make_layer(feature_dims[2], stride=2, dilation=1)

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1, dilation=dilation)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        layer2 = self.layer2(x)  # 1/4

        x = self.layer3(layer2)  # 1/8

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
