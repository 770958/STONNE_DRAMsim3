import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import  os
cfg = ' '
conv_tile = '/mnt/users/wangjs/stonne/benchmarks/image_classification/resnet/tiles/conv_'
fc_tile = '/mnt/users/wangjs/stonne/benchmarks/image_classification/resnet/tiles/fc.txt'
sparsity_ratio = 0.9
output_path = '/mnt/users/wangjs/stonne/benchmarks/image_classification/resnet/result'

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.SimulatedConv2d(in_planes, out_planes, 3, path_to_arch_file=cfg,path_to_tile=conv_tile+'3.txt', sparsity_ratio=sparsity_ratio, groups=1, stride=stride,
                              padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # inplanes代表输入通道数，planes代表输出通道数。
        super(BasicBlock, self).__init__()
        # Conv1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Conv2
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # 下采样
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # F(x)+x
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # layers=参数列表 block选择不同的类
        self.inplanes = 64
        super(ResNet, self).__init__()
        if (output_path != ''):
            os.environ['OUTPUT_DIR'] = output_path
        # 1.conv1
        self.conv1 = nn.SimulatedConv2d(3, 64, 7, path_to_arch_file=cfg,path_to_tile=conv_tile+'7.txt', sparsity_ratio=sparsity_ratio, stride=2, groups=1, padding=3,
                                        bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 2.conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 3.conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 4.conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 5.conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.SimulatedLinear(512 * block.expansion, num_classes, path_to_arch_file=cfg,path_to_tile=fc_tile, sparsity_ratio=sparsity_ratio)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.SimulatedConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1,  path_to_arch_file=cfg,path_to_tile=conv_tile+'1.txt', sparsity_ratio=sparsity_ratio, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 每个blocks的第一个residual结构保存在layers列表中。
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x


def resnet_model(simulation_file=''):
    global cfg
    cfg = simulation_file
    net = ResNet(BasicBlock, [2, 2, 2, 2])
    input_test = torch.randn(3, 224, 224).view(-1, 3, 224, 224)
    output = net(input_test)
