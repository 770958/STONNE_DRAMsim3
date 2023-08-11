from torch import nn
import torch
import os

cfg = ' '
conv_tile = '/mnt/users/wangjs/stonne/benchmarks/image_classification/mobilenetsv2/tiles/tile_configuration_'
sparsity_ratio = 0.9
output_path = '/mnt/users/wangjs/stonne/benchmarks/image_classification/mobilenetsv2/result'


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    """
    定义卷积、批量归一化和激活函数
    group=1表示是普通卷积
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        if out_channel == 24:
            path_to_tile_str = conv_tile + 'conv2.txt'
        elif in_channel == 3:
            path_to_tile_str = conv_tile + 'conv3.txt' # the first CONV
        elif in_channel == 320:
            path_to_tile_str = conv_tile + 'conv4.txt' # the last CONV
        else:
            path_to_tile_str = conv_tile + 'conv1.txt'

        super(ConvBNReLU, self).__init__(
            nn.SimulatedConv2d(in_channel, out_channel, kernel_size, path_to_arch_file=cfg,
                               path_to_tile=path_to_tile_str, stride=stride, padding=padding, groups=groups,
                               bias=False, sparsity_ratio=sparsity_ratio),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        # 表示通过1×1卷积对通道进行扩张。
        hidden_channel = in_channel * expand_ratio
        # 当步长等于1，且输入通道等于输出通道时，使用捷径分支
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 这里判断是因为第一个bottleneck的t为1，也就是并没有使用1×1的卷积进行升维
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        # extend()可以进行多层操作
        layers.extend([
            # 3x3 depthwise conv，通道不发生改变
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            # 最后一个1×1卷积使用的是线性激活函数，所以直接省略即可
            nn.SimulatedConv2d(hidden_channel, out_channel, 1, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv5.txt',
                               sparsity_ratio=sparsity_ratio, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        if (output_path != ''):
            os.environ['OUTPUT_DIR'] = output_path
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)  # 32
        last_channel = _make_divisible(1280 * alpha, round_nearest)  # 1280

        # 倒残差结构参数设置
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # 添加普通的conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            # output_channel
            # 16 24 32 64 96 160 320
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                # 只有第一次时步长为s，后面重复步长都为1
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))  # [320, 1280]
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.SimulatedLinear(last_channel, num_classes, path_to_arch_file=cfg, path_to_tile=conv_tile + 'fc6.txt',
                               sparsity_ratio=sparsity_ratio)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.SimulatedConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.SimulatedLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenets_model(simulation_file=''):
    global cfg
    cfg = simulation_file
    model = MobileNetV2()
    input_test = torch.randn(3, 32, 32).view(-1, 3, 32, 32)
    output = model(input_test)
