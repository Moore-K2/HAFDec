from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
# 尝试导入 Lite-Mono 的 layers，如果是在独立环境测试，则使用下面的 Mock 类
try:
    from layers import *
    from timm.models.layers import trunc_normal_
except ImportError:
    # --- Mock layers for standalone testing ---
    print("Warning: 'layers' module not found. Using mock layers for demonstration.")


    def trunc_normal_(tensor, std=0.02):
        nn.init.normal_(tensor, std=std)


    def upsample(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


    class Conv3x3(nn.Module):
        def __init__(self, in_channels, out_channels, use_refl=True):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        def forward(self, x): return self.conv(x)


    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = Conv3x3(in_channels, out_channels)
            self.nonlin = nn.ELU(inplace=True)

        def forward(self, x): return self.nonlin(self.conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class HFSA(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(HFSA, self).__init__()
        self.in_channel = high_feature_channel + low_feature_channels
        self.out_channel = high_feature_channel if output_channel is None else output_channel
        reduction = 16

        # 直接初始化两个注意力模块
        self.channel_att = ChannelAttention(self.in_channel, reduction)
        self.spatial_att = SpatialAttention()

        self.conv_se = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def upsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, high_features, low_features):
        # 特征对齐 + 拼接
        high_up = self.upsample(high_features, size=low_features.shape[2:])
        features = torch.cat([high_up, low_features], dim=1)

        # 直接调用通道+空间注意力（无封装CBAM类）
        features = features * self.channel_att(features)
        features = features * self.spatial_att(features)

        # 卷积输出
        out = self.relu(self.conv_se(features))
        return out

# ===================== 计算参数量 =====================
def count_params(model, is_print=True):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_print:
        print(f'Total number of parameters: {params}')
    else:
        return params


class HAFDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(HAFDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales
        self.num_ch_enc = num_ch_enc  # features in encoder, [24, 48, 80, 128]

        # decoder
        self.convs = OrderedDict()

        self.convs[("conv3x3", 00)] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])

        self.convs[("conv3x3", 1_0)] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("conv3x3", 2_0)] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])

        self.convs[("conv1x1", 1_1)] = ConvBlock1x1(self.num_ch_enc[2] + self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("conv3x3", 1_2)] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])

        self.convs[("attention", 2_1)] = HFSA(self.num_ch_enc[2], self.num_ch_enc[3])
        self.convs[("conv3x3", 2_2)] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])  # 80->80

        self.convs[("attention", 1_3)] = HFSA(self.num_ch_enc[1], self.num_ch_enc[2])
        self.convs[("conv3x3", 1_4)] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])

        self.convs[("refine_conv", 0)] = ConvBlock(72, 24)

        self.convs[("dispconv", 0)] = Conv3x3(24, self.num_output_channels)
        self.convs[("dispconv", 1)] = Conv3x3(self.num_ch_enc[1], self.num_output_channels)
        self.convs[("dispconv", 2)] = Conv3x3(self.num_ch_enc[2], self.num_output_channels)


        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # e = updown_sample(input_features[0], 2) # up_down_sample表示根据scale进行上采样或下采样
        e0 = input_features[0]  # h/2
        e1 = input_features[1]  # h/4
        e2 = input_features[2]  # h/8
        e3 = input_features[3] # h/16

        d2_0 = self.convs[("conv3x3", 2_0)](e2)
        d1_0 = self.convs[("conv3x3", 1_0)](e1)
        d0_0 = self.convs[("conv3x3", 00)] (e0)

        d2_1 = self.convs[("attention", 2_1)](e3, d2_0)
        d2_2 = self.convs[("conv3x3", 2_2)](d2_1)

        d1_1 = self.FusionConv(self.convs[("conv1x1", 1_1)], d2_0, d1_0)
        d1_2 = self.convs[("conv3x3", 1_2)](d1_1)

        d1_3 = self.convs[("attention", 1_3)](d2_2, d1_2)
        d1_4 = self.convs[("conv3x3", 1_4)](d1_3)

        d0_1 = torch.cat([updown_sample(d1_4, 2), d0_0], 1)
        d0_2 = self.convs[("refine_conv", 0)](d0_1)

        self.outputs[("disp", 0)] = self.sigmoid(updown_sample(self.convs[("dispconv", 0)](d0_2), 2))
        self.outputs[("disp", 1)] = self.sigmoid(updown_sample(self.convs[("dispconv", 1)](d1_4), 2))
        self.outputs[("disp", 2)] = self.sigmoid(updown_sample(self.convs[("dispconv", 2)](d2_2), 2))

        return self.outputs  # single-scale depth


if __name__ == '__main__':
    pass