from torch import nn
import torch

from models.convLSTM import *

class MLConvLSTM(nn.Module):

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, num_layer, stride=1, padding=1, only_last=True):
        super().__init__()


        kernel_size = self._extend_for_multilayer(kernel_size, num_layer)
        num_filter = self._extend_for_multilayer(num_filter, num_layer)
        stride = self._extend_for_multilayer(stride, num_layer)
        padding = self._extend_for_multilayer(padding, num_layer)
        self.num_layer = num_layer
        self.only_last = only_last

        if not len(kernel_size) == len(num_filter) == num_layer:
            raise  ValueError("Not Match layer param length")

        layer_list = []

        for i in range(0, num_layer):
            cur_input_channel = input_channel if i == 0 else num_filter[i-1]

            layer_list.append(ConvLSTM(
                input_channel=cur_input_channel,
                num_filter=num_filter[i],
                b_h_w=b_h_w,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
            ))

        self.layer_list = nn.ModuleList(layer_list)

    def forward(self, inputs=None, states=None, device='cuda'):
        layer_output_list = []
        layer_state_list = []
        cur_layer_input = inputs

        for layer_idx in range(self.num_layer):
            output, state = self.layer_list[layer_idx](inputs=cur_layer_input, device=device)
            cur_layer_input = output

            layer_output_list.append(output)
            layer_state_list.append(state)

        if self.only_last:
            layer_output_list = layer_output_list[-1]
            layer_state_list = layer_state_list[-1]

        # print(layer_output_list.shape)
        return layer_output_list, layer_state_list


class MLBiConvLSTM(nn.Module):

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def __init__(self, input_channel, num_filter, output_channel, b_h_w, kernel_size, num_layer, stride=1, padding=1):
        super().__init__()

        num_filter = self._extend_for_multilayer(num_filter, num_layer)
        for i in range(len(num_filter)):
            num_filter[i] = num_filter[i] //2

        self.forward_net = MLConvLSTM(
            input_channel=input_channel,
            num_filter=num_filter,
            b_h_w=b_h_w,
            num_layer=num_layer,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            only_last=True,
            )

        self.backward_net = MLConvLSTM(
            input_channel=input_channel,
            num_filter=num_filter,
            b_h_w=b_h_w,
            num_layer=num_layer,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            only_last=True,
            )

        last_hidden = num_filter*2 if len(num_filter)<1 else num_filter[-1]*2
        print("Last hidden! ", last_hidden)

        self.conv11 = nn.Conv2d(in_channels=last_hidden, out_channels=output_channel,kernel_size=(1, 1), stride=1)


    def forward(self, forward_x, device='cuda'):

        backward_x = self._make_reverse(forward_x)

        out_fwd, fs = self.forward_net(inputs=forward_x, device=device)
        out_bwd, bs = self.backward_net(inputs=backward_x, device=device)

        # print("=======")
        # t b c h w
        # print(out_fwd.shape)
        reversed_idx = list(reversed(range(out_bwd.shape[1])))
        out_bwd = out_bwd[:,reversed_idx, ...]  # reverse temporal outputs.


        ycat = torch.cat((out_fwd, out_bwd), dim=2)
        # print(ycat.shape)
        b, t, c, h, w = ycat.shape
        ycat = ycat.view(-1, c, h ,w) # B*T C H W


        ycat = self.conv11(ycat)
        ycat = ycat.view(-1, t, ycat.shape[1], h, w)

        # print(ycat.shape)
        # print("==")
        # ycat = ycat.permute(0,2,1,3,4) # B C T H W
        # print(ycat.shape)
        # ycat = self.bn(ycat)
        # ycat = ycat.permute(0, 2, 1, 3, 4) # B T C H W
        # print(ycat.shape)

        ## BN is necessary?

        return ycat, fs



