import torch
import torch.nn as nn

import  os

        # in_channels: int,
        # out_channels: int,
        # kernel_size: _size_2_t,
        # path_to_arch_file: str = '',
        # path_to_tile: str = '',
        # sparsity_ratio: float = 0,
        # stride: _size_2_t = 1,
        # padding: _size_2_t = 0,
        # dilation: _size_2_t = 1,
        # groups: int = 1,
        # bias: bool = True,
        # padding_mode: str = 'zeros'  # TODO: refine this type

cfg = ' '
conv_tile = '/mnt/users/wangjs/stonne/benchmarks/image_classification/yolo4-tiny/tiles/conv_'
sparsity_ratio = 0.9
output_path = '/mnt/users/wangjs/stonne/benchmarks/image_classification/yolo4-tiny/result'

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        if kernel_size==3:
            if in_channels%2==0:
                index=2
            else :
                index=1
        else:
            if out_channels%2==0:
                index=16
            else :
                index=15
        path = conv_tile+str(kernel_size)+'_'+str(index)+'.txt'
        self.conv = nn.SimulatedConv2d(in_channels, out_channels, kernel_size,path_to_arch_file=cfg,path_to_tile=path, sparsity_ratio=sparsity_ratio,stride=stride, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, activation):
        super().__init__()
        hidden_channels = out_channels // 2
        self.downsample = Conv_Bn_Activation(in_channels, out_channels, 3, 2, activation)
        self.split = nn.Sequential(Conv_Bn_Activation(out_channels, hidden_channels, 1, 1, activation),
                                   Conv_Bn_Activation(hidden_channels, hidden_channels, 1, 1, activation))
        self.blocks = nn.Sequential(*[Conv_Bn_Activation(hidden_channels, hidden_channels, 3, 1, activation) for _ in range(num_blocks)])
        self.concat = Conv_Bn_Activation(hidden_channels * 2, out_channels, 1, 1, activation)

    def forward(self, x):
        x = self.downsample(x)
        x1 = self.split(x)
        x2 = self.blocks(x1)
        x = torch.cat((x1, x2), 1)
        return self.concat(x)

class YOLOv4Tiny(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        if (output_path != ''):
            os.environ['OUTPUT_DIR'] = output_path
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.stem = nn.Sequential(Conv_Bn_Activation(3, 32, 3, 2, self.activation),
                                  Conv_Bn_Activation(32, 64, 3, 2, self.activation))
        self.csp_blocks = nn.Sequential(CSPBlock(64, 64, 1, self.activation),
                                        CSPBlock(64, 128, 2, self.activation))
        self.neck = nn.Sequential(Conv_Bn_Activation(128, 256, 3, 2, self.activation),
                                  Conv_Bn_Activation(256, 512, 3, 2, self.activation))
        self.head1 = nn.Sequential(Conv_Bn_Activation(512, 256, 1, 1, self.activation),
                                   Conv_Bn_Activation(256, 512, 3, 1, self.activation),
                                   nn.SimulatedConv2d(512, 3 * (num_classes + 5), 1,  path_to_arch_file=cfg,path_to_tile=conv_tile + '1_15.txt', sparsity_ratio=sparsity_ratio))
        self.head2 = nn.Sequential(Conv_Bn_Activation(256, 128, 1, 1, self.activation),
                                   nn.Upsample(scale_factor=2, mode='nearest'),
                                   Conv_Bn_Activation(128, 256, 3, 1, self.activation),
                                   nn.SimulatedConv2d(256, 3 * (num_classes + 5), 1, path_to_arch_file=cfg,path_to_tile=conv_tile + '1_15.txt', sparsity_ratio=sparsity_ratio))

    def forward(self, x):
        x = self.stem(x)
        x = self.csp_blocks(x)
        x1 = self.neck(x)
        out1 = self.head1(x1)
        x2 = self.head2(x1)
        return out1, x2

def yolo_model(simulation_file=''):
    global cfg
    cfg = simulation_file
    model = YOLOv4Tiny()
    input_test = torch.randn(1, 3, 416, 416) # 1张3通道416x416的图片
    output = model(input_test)