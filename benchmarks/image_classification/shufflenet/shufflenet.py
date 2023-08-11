import torch
import torch.nn as nn
from torch.utils import data

import os
cfg = ' '
conv_tile = '/mnt/users/wangjs/stonne/benchmarks/image_classification/shufflenet/tiles/tile_configuration_'
sparsity_ratio = 0.9
output_path = '/mnt/users/wangjs/stonne/benchmarks/image_classification/shufflenet/result'


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    # 1x1卷积操作
    return nn.SimulatedConv2d(in_channels, out_channels, 1, path_to_arch_file=cfg,path_to_tile=conv_tile, sparsity_ratio=sparsity_ratio,
                              stride=stride, groups=groups, bias=bias )


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1, groups=1, bias=False):
    # 3x3卷积操作
    # 默认不是下采样
    return nn.SimulatedConv2d(in_channels, out_channels, 3, path_to_arch_file=cfg,path_to_tile=conv_tile, sparsity_ratio=sparsity_ratio,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)


def depthwise_con3x3(channels, stride):
    # 空间特征抽取
    # 输入通道和输出通道相等，且分组数等于通道数
    return nn.SimulatedConv2d(channels, channels, 3, path_to_arch_file=cfg,path_to_tile=conv_tile, sparsity_ratio=sparsity_ratio,
                              stride=stride, padding=1, groups=channels, bias=False )


def channel_shuffle(x, groups):
    # x[batch_size, channels, H, W]
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups  # 每组通道数
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle(nn.Module):
    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        if channels % groups != 0:
            raise ValueError("通道数必须可以整除组数")
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups, downsample, ignore_group):
        # 如果做下采样，那么通道数翻倍，高宽减半
        # 如果不做下采样，那么输入输出通道数相等，高宽不变
        super(ShuffleUnit, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4

        if downsample:
            out_channels -= in_channels
        else:
            assert in_channels == out_channels, "不做下采样时应该输入输出通道相等"

        self.compress_conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=(1 if ignore_group else groups)
        )

        self.compress_bn1 = nn.BatchNorm2d(num_features=mid_channels)

        self.c_shuffle = ChannelShuffle(channels=mid_channels, groups=groups)

        self.dw_conv2 = depthwise_con3x3(channels=mid_channels, stride=(2 if downsample else 1))

        self.dw_bn2 = nn.BatchNorm2d(num_features=mid_channels)

        self.expand_conv3 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            groups=groups
        )

        self.expand_bn3 = nn.BatchNorm2d(num_features=out_channels)

        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.compress_conv1(x)  # x[batch_size, mid_channels, H, W]
        x = self.compress_bn1(x)
        x = self.activ(x)
        x = self.c_shuffle(x)
        x = self.dw_conv2(x)  # x[batch_size, mid_channels, H, w]
        x = self.dw_bn2(x)
        x = self.expand_conv3(x)  # x[batch_size, out_channels, H, W]
        x = self.expand_bn3(x)
        if self.downsample:
            identity = self.avgpool(identity)
            x = torch.cat((x, identity), dim=1)  # 通道维上拼接
        else:
            x = x + identity
        x = self.activ(x)
        return x


class ShuffleInitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShuffleInitBlock, self).__init__()
        self.conv = nn.SimulatedConv2d(in_channels, out_channels, 3, stride=2, padding=1,
                                       path_to_arch_file=cfg,path_to_tile=conv_tile, sparsity_ratio=sparsity_ratio)  # 下采样
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 下采样

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class ShuffleNet(nn.Module):
    def __init__(self, channels, init_block_channels, groups, in_channels=1, in_size=(224, 224), num_classes=10):
        super(ShuffleNet, self).__init__()
        if (output_path != ''):
            os.environ['OUTPUT_DIR'] = output_path
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ShuffleInitBlock(in_channels, init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                downsample = (j == 0)
                ignore_group = (i == 0) and (j == 0)
                stage.add_module("unit{}".format(j + 1), ShuffleUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=groups,
                    downsample=downsample,
                    ignore_group=ignore_group))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.SimulatedLinear(in_channels, num_classes, path_to_arch_file=cfg,path_to_tile=fc_tile, sparsity_ratio=sparsity_ratio)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_shufflenet(groups, width_scale):
    init_block_channels = 24
    layers = [2, 4, 2]
    if groups == 1:
        channels_per_layers = [144, 288, 576]
    elif groups == 2:
        channels_per_layers = [200, 400, 800]
    elif groups == 3:
        channels_per_layers = [240, 480, 960]
    elif groups == 4:
        channels_per_layers = [272, 544, 1088]
    elif groups == 8:
        channels_per_layers = [384, 768, 1536]
    else:
        raise ValueError("The {} of groups is not supported".format(groups))
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)

    net = ShuffleNet(
        channels=channels,
        init_block_channels=init_block_channels,
        groups=groups)

    return net


def shufflenet_model(simulation_file=''):
    global cfg
    cfg = simulation_file
    model = get_shufflenet(1, 1.0)
    input_test = torch.randn(1, 224, 224).view(-1, 1, 224, 224)
    output = model(input_test)
