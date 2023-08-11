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

    print(channels,init_block_channels,groups)

get_shufflenet(1, 1.0)