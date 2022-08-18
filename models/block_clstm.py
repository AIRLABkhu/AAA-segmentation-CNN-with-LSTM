import torch.nn as nn
import torch.nn.functional as F
from .convLSTM import *
from .m2oLSTM import *


class resConv(nn.Module):

    def __init__(self,in_channels, conv_channel, num_layer=1):
        super().__init__()

        self.num_later=num_layer
        block = []

        for i in range(num_layer):
            block.append(nn.Conv2d(in_channels=in_channels, out_channels=conv_channel, kernel_size=3, padding=1, stride=1))
            block.append(nn.BatchNorm2d(conv_channel))
            if i < num_layer-1:
                block.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv_block = nn.Sequential(*block)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
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
        out = self.lrelu(out)

        return out

class block_LSTM(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h ,w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, in_channels, out_channels, num_filter, num_conv_layer=1):
        super().__init__()

        self.conv = resConv(in_channels=in_channels, conv_channel=num_filter, num_layer=num_conv_layer)

        self.bi_clstm = BiConvLSTM(input_channel=num_filter, num_filter=num_filter, output_channel=out_channels,
                                kernel_size=(3,3), padding=1, b_h_w=(1, 512, 512))

    def forward(self,x, device):

        output, t = self.voltoslices(x)
        output = self.conv(output)
        output = self.slicetovol(output, t)

        output, state = self.bi_clstm(output, device=device)
        return output, state

class block_m2o_LSTM(nn.Module):

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, in_channels, num_filter, m_length, num_conv_layer=1, out_channels=1):
        super().__init__()

        self.conv = resConv(in_channels=in_channels, conv_channel=num_filter, num_layer=num_conv_layer)

        self.bi_clstm = m2oBiConvLSTM(input_channel=num_filter, num_filter=num_filter, output_channel=out_channels,
                                kernel_size=(3,3), m_length=m_length, padding=1, b_h_w=(1, 512, 512))

    def forward(self,x, device):

        output, t = self.voltoslices(x)
        output = self.conv(output)
        output = self.slicetovol(output, t)

        output, state = self.bi_clstm(output, device=device)
        return output, state

class multi_block_LSTM(nn.Module):

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def __init__(self, in_channels, out_channels, m_length,
                 num_filter, num_block, num_conv_layer=1):
        super().__init__()

        self.num_block = num_block

        num_filter = self._extend_for_multilayer(num_filter, num_block)
        if not len(num_filter) == num_block:
            raise  ValueError("Not Match layer param length")

        cur_input_channel = in_channels

        blocks = []

        for i in range(num_block-1):
            blocks.append(block_LSTM(in_channels=cur_input_channel, out_channels=num_filter[i], num_filter=num_filter[i], num_conv_layer=num_conv_layer))
            cur_input_channel = num_filter[i]

        blocks.append(block_m2o_LSTM(in_channels=cur_input_channel, out_channels=out_channels, num_filter=num_filter[-1],
                                     m_length=m_length, num_conv_layer=num_conv_layer))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, device):

        cur_layer_input = x
        for layer_idx  in range(self.num_block):
            output, state = self.blocks[layer_idx](cur_layer_input, device)
            cur_layer_input = output

        return output, state