""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet3d_parts import *


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, out_channels=64, mid_channels=32)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)

        x = self.up1(x4, x3)
        # print(x.shape)
        x = self.up2(x, x2)
        # print(x.shape)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return logits
