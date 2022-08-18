""" Full assembly of the parts to form the complete network """

import torch.nn as nn

from .unet_parts import *
from models.convLSTM import BiConvLSTM
from models.m2oLSTM import m2oBiConvLSTM


class Senosr3D(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, n_channels, n_classes, b_h_w, bilinear=True, m_length=2):
        super(Senosr3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        lstmb_h_w = b_h_w[0], b_h_w[1] // 8, b_h_w[2] // 8


        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = nn.MaxPool2d(2)

        self.bidir1 = BiConvLSTM(256, 512, 512, b_h_w=lstmb_h_w,kernel_size=(3,3)) # 3x256x16x16 => 3x512x16x16

        self.up1 = Up(512+256, 256, bilinear)
        self.up2 = Up(256+128, 128, bilinear)
        self.up3 = Up(128+64, 64, bilinear)  # double conv

        self.bidir2 = m2oBiConvLSTM(64,64,64,b_h_w=b_h_w, kernel_size=(3,3), m_length=m_length)

        self.out1 = OutConv(64, 1)

    def forward(self, x, device):

        x, t = self.voltoslices(x)

        x1 = self.inc(x)            #64
        x2 = self.down1(x1)         #128
        x3 = self.down2(x2)         #256
        x3 = self.down3(x3)

        x4 = self.slicetovol(x3,t)
        x4, _ = self.bidir1(x4, device)
        x4, t = self.voltoslices(x4)    #512

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.slicetovol(x, t)
        x, _ = self.bidir2(x, device)
        x, t = self.voltoslices(x)

        logits = self.out1(x)

        logits = self.slicetovol(logits, t)
        return logits
