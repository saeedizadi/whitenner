# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F
import logging


class UNet(nn.Module):
    def __init__(self, in_channels=1, depth=2, act='relu', wf=6, 
                 padding=True, batch_norm=False, up_mode='upsample', 
                 twohead=False, _log=logging.getLogger(), **kwargs):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()

        self.padding = padding
        self.twohead = twohead
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i),
                              padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.latent_signal = nn.Conv2d(prev_channels, prev_channels, kernel_size=1)
        self.decoder_signal = UNetDecoder(depth, prev_channels, up_mode, padding, batch_norm, wf, 1)        
        
        if self.twohead:
            self.latent_noise = nn.Conv2d(prev_channels, prev_channels, kernel_size=1)
            self.decoder_noise = UNetDecoder(depth, prev_channels, up_mode, padding, batch_norm, wf, 1)

        

        _log.info("Created a UNet with the following properties:\nDepth: {}  Act: {}  WF: {}  BatchNorm: {}  UpMode: {}".format(depth, act, wf, batch_norm, up_mode))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
      

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        ls = self.latent_signal(x)
        signal = self.decoder_signal(ls, blocks)
        out = signal

        if self.twohead:
            ln = self.latent_noise(x)
            noise = self.decoder_noise(ln, blocks)
            out = (signal, noise)


        return out

class UNetDecoder(nn.Module):
    def __init__(self, depth, prev_channels, up_mode, padding, batch_norm, wf, n_classes):
        super(UNetDecoder, self).__init__()

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i),
                            up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
        
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x, blocks):
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        out = self.last(x)
        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size,
                               kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size,
                               kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(
                in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
