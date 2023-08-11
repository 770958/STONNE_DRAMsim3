import torch
import torch.nn as nn
from utils import load_state_dict_from_url
import os

cfg = ""
conv_tile = '/mnt/users/wangjs/stonne/benchmarks/image_classification/alexnet/tiles/tile_configuration_'
fc_tile = '/mnt/users/wangjs/stonne/benchmarks/image_classification/alexnet/tiles/tile_configuration_'
sparsity_ratio = 0.9
output_path = '/mnt/users/wangjs/stonne/benchmarks/image_classification/alexnet/result'


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        if (output_path != ''):
            os.environ['OUTPUT_DIR'] = output_path

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.features = nn.Sequential(
            nn.SimulatedConv2d(3, 64, kernel_size=11,
                               path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv1.txt',
                               sparsity_ratio=sparsity_ratio, stride=4,
                               padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.SimulatedConv2d(64, 192, kernel_size=5,
                               path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv2.txt',
                               sparsity_ratio=sparsity_ratio, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.SimulatedConv2d(192, 384, kernel_size=3,
                               path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv3.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
            nn.ReLU(inplace=True),
            nn.SimulatedConv2d(384, 256, kernel_size=3,
                               path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv4.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
            nn.ReLU(inplace=True),
            nn.SimulatedConv2d(256, 256, kernel_size=3,
                               path_to_arch_file=cfg, path_to_tile=conv_tile + 'conv5.txt',
                               sparsity_ratio=sparsity_ratio, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.SimulatedLinear(256 * 6 * 6, 4096,
                               path_to_arch_file=cfg, path_to_tile=fc_tile + 'fc6.txt',
                               sparsity_ratio=sparsity_ratio, ),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.SimulatedLinear(4096, 4096,
                               path_to_arch_file=cfg, path_to_tile=fc_tile + 'fc7.txt',
                               sparsity_ratio=sparsity_ratio, ),
            nn.ReLU(inplace=True),
            nn.SimulatedLinear(4096, num_classes,
                               path_to_arch_file=cfg, path_to_tile=fc_tile + 'fc8.txt',
                               sparsity_ratio=sparsity_ratio, ),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet_model(simulation_file=''):
    global cfg
    cfg = simulation_file
    model = AlexNet()
    # state_dict = load_state_dict_from_url(model_urls['alexnet'],progress=progress)
    state_dict = torch.load("/mnt/users/wangjs/stonne/benchmarks/image_classification/alexnet/alexnet.pth")
    model.load_state_dict(state_dict)
    return model

# alex_model = alexnet(pretrained=True)
