import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os
cfg = ' '
conv_tile = '/mnt/users/wangjs/stonne/benchmarks/image_classification/vgg/tiles/tile_configuration_'
sparsity_ratio = 0.9
output_path = '/mnt/users/wangjs/stonne/benchmarks/image_classification/vgg/result'


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        if (output_path != ''):
            os.environ['OUTPUT_DIR'] = output_path
        self.features = nn.Sequential(
                    nn.SimulatedConv2d(3, 64, kernel_size=3, path_to_arch_file=cfg,path_to_tile=conv_tile + 'conv1.txt',
                               sparsity_ratio=sparsity_ratio,padding=1),
                    nn.ReLU(inplace=True),
                    nn.SimulatedConv2d(64, 64, kernel_size=3, path_to_arch_file=cfg,path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio,padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.SimulatedConv2d(64, 128, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.SimulatedConv2d(128, 128, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.SimulatedConv2d(128, 256, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.SimulatedConv2d(256, 256, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.SimulatedConv2d(256, 256, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.SimulatedConv2d(256, 512, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.SimulatedConv2d(512, 512, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.SimulatedConv2d(512, 512, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.SimulatedConv2d(512, 512, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.SimulatedConv2d(512, 512, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.SimulatedConv2d(512, 512, kernel_size=3, path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
         )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.SimulatedLinear(512 * 7 * 7, 4096,path_to_arch_file=cfg,path_to_tile=conv_tile + 'fc6.txt', sparsity_ratio=sparsity_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.SimulatedLinear(4096, 4096,path_to_arch_file=cfg,path_to_tile=conv_tile + 'fc6.txt', sparsity_ratio=sparsity_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.SimulatedLinear(4096, num_classes,path_to_arch_file=cfg,path_to_tile=conv_tile + 'fc7.txt', sparsity_ratio=sparsity_ratio),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg16_model(simulation_file=''):
    global cfg
    cfg = simulation_file
    model = VGG16()
    state_dict = torch.load("/mnt/users/wangjs/stonne/benchmarks/image_classification/vgg/vgg16.pth")
    model.load_state_dict(state_dict)
    return model
