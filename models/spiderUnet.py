""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
from models.convLSTM import BiConvLSTM


class SpiderUnet(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, n_channels, n_classes, b_h_w, bilinear=False):
        super(SpiderUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        lstmb_h_w = b_h_w[0], b_h_w[1] // 16, b_h_w[2] // 16
        # print(b_h_w)
        # print(lstmb_h_w)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 196)
        self.down3 = Down(196, 256)
        self.down4 = Down(256, 512)

        self.lstm_layer = BiConvLSTM(512, 512, 512, b_h_w=lstmb_h_w,kernel_size=(3,3))

        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 196, bilinear)
        self.up3 = Up(196, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out1 = OutConv(64, 32)
        self.out2 = OutConv(32, n_channels)

    def forward(self, x, device):

        x, t = self.voltoslices(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.slicetovol(x5,t)
        x5, _ = self.lstm_layer(x5, device)
        x5, t = self.voltoslices(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out1(x)
        logits = self.out2(x)

        logits = self.slicetovol(logits, t)
        return logits
