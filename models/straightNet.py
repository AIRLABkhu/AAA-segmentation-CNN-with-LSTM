import torch.nn as nn
import torch.nn.functional as F
from .convLSTM import BiConvLSTM
# from .MLLSTM import *

class straightConv(nn.Module):

    def __init__(self, in_channels, conv_channel):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv_channel, kernel_size=(3,3), padding=1, stride=1),
            nn.BatchNorm2d(conv_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_channel, out_channels=conv_channel, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(conv_channel)
        )
        self.relu = nn.ReLU(inplace=True)
        self.ch_change = False
        if in_channels != conv_channel:
            self.ch_change = True
            self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=conv_channel, kernel_size=(1,1))

    def forward(self, x):
        identity = x

        out = self.conv_block(x)

        if self.ch_change:
            identity = self.conv11(identity)

        out += identity
        out = self.relu(out)

        return out

class straightNet(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, in_channels, conv_channel, out_channels, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=out_channels)
        self.fuse = nn.Conv2d(5,1,1)

    def forward(self, x):
        # print(x.shape)
        output, t = self.voltoslices(x)
        # print(output.shape, t)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)
        output = output.squeeze(2)
        if t < 5:
            diff = 5 -t
            pad = (0,0, 0,0, 0,diff)
            output = F.pad(output, pad, "constant", 0)

        output = self.fuse(output)
        return output

class straightNetBack(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h ,w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self,in_channels, conv_channel, lstm_channel, HALF_MODE=False):
        super().__init__()


        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w=(1, 256, 256)
        else:
            b_h_w=(1, 512, 512)

        self.bi_convlstm = BiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, output_channel=1, kernel_size=(3,3), padding=1, b_h_w=b_h_w)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.bi_convlstm(output, device=device)
        return output, state


class straightNetMid(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape

        return input.view(-1, c, h ,w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self,in_channels, conv_channel, lstm_channel, HALF_MODE=False):
        super().__init__()
        if HALF_MODE:
            b_h_w=(1, 256, 256)
        else:
            b_h_w=(1, 512, 512)

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.bi_convlstm = BiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, output_channel=lstm_channel, kernel_size=(3,3), padding=1, b_h_w=b_h_w)
        self.conv_block2 = straightConv(in_channels=lstm_channel, conv_channel=1)


    def forward(self, x, device):

        output, t = self.voltoslices(x)
        output = self.conv_block1(output)
        output = self.slicetovol(output, t)

        output, state = self.bi_convlstm(output, device=device)

        output, t = self.voltoslices(output)
        output = self.conv_block2(output)
        output = self.slicetovol(output, t)

        return output, state



class straightNetFrt(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape

        return input.view(-1, c, h ,w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self,in_channels, conv_channel, lstm_channel, HALF_MODE=False):
        super().__init__()
        if HALF_MODE:
            b_h_w=(1, 256, 256)
        else:
            b_h_w=(1, 512, 512)

        self.bi_convlstm = BiConvLSTM(input_channel=in_channels, num_filter=lstm_channel, output_channel=lstm_channel,
                                      kernel_size=(3, 3), padding=1, b_h_w=b_h_w)
        self.conv_block1 = straightConv(in_channels=lstm_channel, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=1)


    def forward(self, x, device):

        output, state = self.bi_convlstm(x, device=device)
        output, t = self.voltoslices(output)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        return output, state


class straightNetBack_lc(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h ,w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self,in_channels, conv_channel, lstm_channel, HALF_MODE=False):
        super().__init__()


        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)
        self.conv_block3 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w=(1, 256, 256)
        else:
            b_h_w=(1, 512, 512)

        self.bi_convlstm = BiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, output_channel=1, kernel_size=(3,3), padding=1, b_h_w=b_h_w)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)
        output = self.conv_block3(output)

        output = self.slicetovol(output, t)

        output, state = self.bi_convlstm(output, device=device)
        return output, state

class straightNetBack_sl(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h ,w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self,in_channels, conv_channel, lstm_channel, HALF_MODE=False):
        super().__init__()


        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w=(1, 256, 256)
        else:
            b_h_w=(1, 512, 512)

        self.bi_convlstm1 = BiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, output_channel=lstm_channel, kernel_size=(3,3), padding=1, b_h_w=b_h_w)
        self.bi_convlstm2 = BiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, output_channel=1, kernel_size=(3,3), padding=1, b_h_w=b_h_w)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.bi_convlstm1(output, device=device)
        output, state = self.bi_convlstm2(output, device=device)

        return output, state


class straightNetBack_ml(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h ,w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self,in_channels, conv_channel, lstm_channel, HALF_MODE=False):
        super().__init__()


        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w=(1, 256, 256)
        else:
            b_h_w=(1, 512, 512)

        # self.ml_biconvlstm = MLBiConvLSTM(input_channel=in_channels, num_filter=lstm_channel, output_channel=1, num_layer=2, kernel_size=(3,3), padding=1, b_h_w=b_h_w)

    def forward(self, x, device):
        # print(x.shape)
        # output, t = self.voltoslices(x)
        # print(output.shape)
        # output = self.conv_block1(output)
        # output = self.conv_block2(output)
        # output = self.slicetovol(output, t)
        # print(output.shape)
        output = self.ml_biconvlstm(x, device=device)
        # print(output.shape)
        return output, x

class straightNetBack_DEEP(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h ,w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self,in_channels, conv_channel, lstm_channel, HALF_MODE=False):
        super().__init__()


        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel) ## 16
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel*2) ## 32
        self.conv_block3 = straightConv(in_channels=conv_channel*2, conv_channel=conv_channel*4) ## 64

        if HALF_MODE:
            b_h_w=(1, 256, 256)
        else:
            b_h_w=(1, 512, 512)

        self.ml_biconvlstm = MLBiConvLSTM(input_channel=conv_channel*4, num_filter=conv_channel*4, output_channel=conv_channel*2, num_layer=2, kernel_size=(3,3), padding=1, b_h_w=b_h_w)
        self.bi_convlstm = BiConvLSTM(input_channel=conv_channel*2, num_filter=conv_channel, output_channel=1, kernel_size=(3,3), padding=1, b_h_w=b_h_w)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)
        output = self.conv_block3(output)

        output = self.slicetovol(output, t)

        output = self.ml_biconvlstm(output, device=device)
        # print(output.shape)
        output, state = self.bi_convlstm(output, device=device)

        return output, state