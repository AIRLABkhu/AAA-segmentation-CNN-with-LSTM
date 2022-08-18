from torch import nn
import torch

"""
Normalization Helps Training of quantized models
"""


class ConvLSTM(nn.Module):
    """
    https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py

    """
    # single conv : output_channel in parameter, conv 11
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        # self._2dbn = nn.BatchNorm2d(num_filter*4)
        self._batch_size, self._state_height, self._state_width = b_h_w
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width, requires_grad=True))
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width, requires_grad=True))
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width, requires_grad=True))
        self._input_channel = input_channel
        self._num_filter = num_filter

        # self.conv11 = nn.Conv2d(in_channels=num_filter,
        #                         out_channels=output_channel,
        #                         kernel_size=(1,1),
        #                         stride=1)

        # self.bn = nn.BatchNorm3d(num_filter)
        group_num = num_filter // 16 ## from GN paper
        if group_num <= 0:
            group_num=1
        self.gn = nn.GroupNorm(group_num,num_filter)
        # self.bn = nn.BatchNorm2d(num_filter)

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W
    def forward(self, inputs=None, states=None, device='cuda'):

        ## B S C H W -> S B C H W
        inputs = inputs.permute(1, 0 ,2, 3, 4)

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(device)
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(device)
        else:
            h, c = states

        outputs = []
        for index in range(inputs.size(0)):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                 self._state_width), dtype=torch.float).to(device)
            else:
                x = inputs[index, ...]

            cat_x = torch.cat([x, h], dim=1)
            # print(cat_x.shape)
            conv_x = self._conv(cat_x)
            # conv_x = self._2dbn(conv_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i + self.Wci * c)
            f = torch.sigmoid(f + self.Wcf * c)
            c = f * c + i * torch.tanh(tmp_c)
            o = torch.sigmoid(o + self.Wco * c)
            h = o * torch.tanh(c)

            # print(h.shape)
            # 2D BN
            # output = self.bn(h)
            # outputs.append(output)

            # 3D BN
            outputs.append(h)

        outputs = torch.stack(outputs)
        outputs = outputs.permute(1,2,0,3,4) # B C T H W
        # 3D BN
        # outputs = self.bn(outputs)
        outputs = self.gn(outputs)

        outputs = outputs.permute(0, 2, 1, 3, 4) # B T C H W

        return outputs, (h,c)

# class BNConvLSTM(nn.Module):
#     """
#     https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py
#     paper " Reccurnet batch normalization "
#
#     """
#     def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1):
#         super().__init__()
#         self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
#                                out_channels=num_filter*4,
#                                kernel_size=kernel_size,
#                                stride=stride,
#                                padding=padding)
#         self._batch_size, self._state_height, self._state_width = b_h_w
#         # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
#         # Howerver, if you use declare an optimizer like Adam(model.parameters()),
#         # parameters will not be updated forever.
#         self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
#         self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
#         self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
#         self._input_channel = input_channel
#         self._num_filter = num_filter
#
#         self.bn_xhh = nn.BatchNorm2d(num_filter*4)
#         self.bn_c = nn.BatchNorm2d(num_filter)
#
#     # inputs and states should not be all none
#     # inputs: S*B*C*H*W => B*S*C*H*W
#     def forward(self, inputs=None, states=None, device='cuda'):
#
#         ## B S C H W -> S B C H W
#         inputs = inputs.permute(1, 0 ,2, 3, 4)
#
#
#         if states is None:
#             c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
#                              self._state_width), dtype=torch.float).to(device)
#             h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
#                              self._state_width), dtype=torch.float).to(device)
#         else:
#             h, c = states
#
#         outputs = []
#         for index in range(inputs.size(0)):
#             # initial inputs
#             if inputs is None:
#                 x = torch.zeros((h.size(0), self._input_channel, self._state_height,
#                                  self._state_width), dtype=torch.float).to(device)
#             else:
#                 x = inputs[index, ...]
#             cat_x = torch.cat([x, h], dim=1)
#             conv_x = self._conv(cat_x)
#             # print("convx ", conv_x.shape)
#             conv_x = self.bn_xhh(conv_x)
#
#             i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)
#
#             i = torch.sigmoid(i + self.Wci * c)
#             f = torch.sigmoid(f + self.Wcf * c)
#
#             c = f * c + i * torch.tanh(tmp_c)
#             # print("c", c.shape)
#             c = self.bn_c(c)
#             o = torch.sigmoid(o + self.Wco * c)
#             h = o * torch.tanh(c)
#             outputs.append(h)
#
#         outputs = torch.stack(outputs)
#         outputs = outputs.permute(1,0,2,3,4) # B T C H W
#
#         return outputs, (h, c)
#
# class LNConvLSTM(nn.Module):
#     """
#     https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py
#
#     """
#     def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1, device='cuda'):
#         super().__init__()
#         self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
#                                out_channels=num_filter*4,
#                                kernel_size=kernel_size,
#                                stride=stride,
#                                padding=padding)
#         self._batch_size, self._state_height, self._state_width = b_h_w
#         # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
#         # Howerver, if you use declare an optimizer like Adam(model.parameters()),
#         # parameters will not be updated forever.
#         self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
#         self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
#         self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
#         self._input_channel = input_channel
#         self._num_filter = num_filter
#
#         self.LN = nn.LayerNorm([input_channel, self._state_height, self._state_width]) # c, h, w
#
#
#     # inputs and states should not be all none
#     # inputs: S*B*C*H*W => B*S*C*H*W
#     def forward(self, inputs=None, states=None, device='cuda'):
#
#         ## B S C H W -> S B C H W
#         inputs = inputs.permute(1, 0 ,2, 3, 4)
#
#
#         if states is None:
#             c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
#                              self._state_width), dtype=torch.float).to(device)
#             h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
#                              self._state_width), dtype=torch.float).to(device)
#         else:
#             h, c = states
#
#         outputs = []
#         for index in range(inputs.size(0)):
#             # initial inputs
#             if inputs is None:
#                 x = torch.zeros((h.size(0), self._input_channel, self._state_height,
#                                  self._state_width), dtype=torch.float).to(device)
#             else:
#                 x = inputs[index, ...]
#
#             x = self.LN(x)
#             cat_x = torch.cat([x, h], dim=1)
#             conv_x = self._conv(cat_x)
#
#             i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)
#
#             i = torch.sigmoid(i + self.Wci * c)
#             f = torch.sigmoid(f + self.Wcf * c)
#             c = f * c + i * torch.tanh(tmp_c)
#             o = torch.sigmoid(o + self.Wco * c)
#             h = o * torch.tanh(c)
#
#             outputs.append(h)
#
#         outputs = torch.stack(outputs)
#         outputs = outputs.permute(1, 0, 2, 3, 4)  # B T C H W
#
#         return outputs, (h, c)


class BiConvLSTM(nn.Module):

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, input_channel, num_filter, output_channel, b_h_w, kernel_size, stride=1, padding=1):
        super(BiConvLSTM, self).__init__()

        self.forward_net = ConvLSTM(input_channel=input_channel, num_filter= num_filter//2, b_h_w=b_h_w,
                                        kernel_size=kernel_size, stride=stride, padding=padding)
        self.backward_net = ConvLSTM(input_channel=input_channel, num_filter= num_filter//2, b_h_w=b_h_w,
                                        kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        #sonsor3D2
        # self.bn = nn.BatchNorm2d(output_channel)

        # biconvlstm NetBack
        self.bn = nn.BatchNorm3d(output_channel)

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):

        backward = self._make_reverse(forward)

        # print(forward.shape)
        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(forward, device=device)
        out_bwd, b_state = self.backward_net(backward, device=device)
        # t b c h w

        out_bwd = self._make_reverse(out_bwd)

        # print("=======")

        ycat = torch.cat((out_fwd, out_bwd), dim=2)
        # print(ycat.shape)
        b, t, c, h, w = ycat.shape
        ycat = ycat.view(-1, c, h ,w) # B*T C H W
        # print(ycat.shape)

        ycat = self.conv11(ycat)
        ycat = ycat.view(-1, t, ycat.shape[1], h, w)
        # print(ycat.shape)
        # ycat = ycat.permute(0,2,1,3,4) # B C T H W
        # print(ycat.shape)
        # ycat = self.bn(ycat)
        # ycat = ycat.permute(0, 2, 1, 3, 4) # B T C H W
        # print(ycat.shape)

        ## BN is necessary?

        return ycat, (f_state)