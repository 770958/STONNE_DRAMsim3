from torch import nn
import torch
import torch.nn.functional as F
import os

cfg = ' '
conv_tile = '/mnt/users/wangjs/stonne/benchmarks/image_classification/lenet/tiles/tile_configuration_'
sparsity_ratio = 0.9
output_path = '/mnt/users/wangjs/stonne/benchmarks/image_classification/lenet/result'


class LeNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(LeNet, self).__init__()
        if (output_path != ''):
            os.environ['OUTPUT_DIR'] = output_path

        self.conv1 = nn.SimulatedConv2d(3, 16, 5, stride=1, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv1.txt',
                                        sparsity_ratio=sparsity_ratio)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.SimulatedConv2d(16, 32, 5, stride=1, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                                        sparsity_ratio=sparsity_ratio)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.SimulatedLinear(32 * 5 * 5, 120, path_to_arch_file=cfg, path_to_tile=conv_tile + 'fc6.txt',
                                      sparsity_ratio=sparsity_ratio)
        self.fc2 = nn.SimulatedLinear(120, 84, path_to_arch_file=cfg, path_to_tile=conv_tile + 'fc7.txt',
                                      sparsity_ratio=sparsity_ratio)
        self.fc3 = nn.SimulatedLinear(84, num_classes, path_to_arch_file=cfg, path_to_tile=conv_tile + 'fc8.txt',
                                      sparsity_ratio=sparsity_ratio)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)  # input(3, 32, 32)  output(16, 28, 28)
        x = self.relu(x)  # 激活函数
        x = self.maxpool1(x)  # output(16, 14, 14)
        x = self.conv2(x)  # output(32, 10, 10)
        x = self.relu(x)  # 激活函数
        x = self.maxpool2(x)  # output(32, 5, 5)
        x = torch.flatten(x, start_dim=1)  # output(32*5*5) N代表batch_size
        x = self.fc1(x)  # output(120)
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # output(84)
        x = self.relu(x)  # 激活函数
        x = self.fc3(x)  # output(num_classes)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def lenet_model(simulation_file=''):
    global cfg
    cfg = simulation_file
    model = LeNet()
    input_test = torch.randn(3, 32, 32).view(-1, 3, 32, 32)
    output = model(input_test)
