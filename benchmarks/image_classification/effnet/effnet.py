import torch.nn as nn
import torch
import os
cfg = '/mnt/users/wangjs/stonne/simulation_files/sigma_128mses_64_bw.cfg'
conv_tile = '/mnt/users/wangjs/stonne/simulation_files/tiles/conv.txt'
fc_tile = '/mnt/users/wangjs/stonne/simulation_files/tiles/fc.txt'
sparsity_ratio = 0.9
output_path = '/mnt/users/wangjs/stonne/benchmarks/image_classification/effnet/result'


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class EffNet(nn.Module):
    def __init__(self, nb_classes=10, include_top=True, weights=None):
        super(EffNet, self).__init__()
        if (output_path != ''):
            os.environ['OUTPUT_DIR'] = output_path
        self.block1 = self.make_layers(32, 64)
        self.block2 = self.make_layers(64, 128)
        self.block3 = self.make_layers(128, 256)
        self.flatten = Flatten()
        self.linear = nn.SimulatedLinear(4096, nb_classes,  bias=False, path_to_arch_file=cfg,path_to_tile=fc_tile, sparsity_ratio=sparsity_ratio)
        self.include_top = include_top
        self.weights = weights

    def make_layers(self, ch_in, ch_out):
        layers = [
            nn.SimulatedConv2d(3, ch_in, (1, 1),   stride=(1, 1), bias=False, padding=0, path_to_arch_file=cfg,path_to_tile=conv_tile, sparsity_ratio=sparsity_ratio,
                               dilation=(1, 1)) if ch_in == 32 else nn.SimulatedConv2d(ch_in, ch_in, kernel_size=(1, 1),
                                                                              stride=(1, 1), bias=False, padding=0,path_to_arch_file=cfg,path_to_tile=conv_tile, sparsity_ratio=sparsity_ratio,
                                                                              dilation=(1, 1)),
            self.make_post(ch_in),
            # DepthWiseConvolution2D
            nn.SimulatedConv2d(ch_in, 1 * ch_in, (1, 3), cfg, conv_tile, stride=(1, 1), groups=ch_in, padding=(0, 1),
                               bias=False, dilation=(1, 1)),
            self.make_post(ch_in),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # DepthWiseConvolution2D
            nn.SimulatedConv2d(ch_in, 1 * ch_in, (3, 1), cfg, conv_tile, groups=ch_in, stride=(1, 1), padding=(1, 0),
                               bias=False, dilation=(1, 1)),
            self.make_post(ch_in),
            nn.SimulatedConv2d(ch_in, ch_out, (1, 2), cfg, conv_tile, stride=(1, 2), bias=False, padding=(0, 0),
                               dilation=(1, 1)),
            self.make_post(ch_out),
        ]
        return nn.Sequential(*layers)

    def make_post(self, ch_in):
        layers = [
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(ch_in, momentum=0.99)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.linear(x)
        return x

def effnet_model():
    model = EffNet()
    input_test = torch.randn(3, 32, 32).view(-1, 3, 32, 32)
    output = model(input_test)
