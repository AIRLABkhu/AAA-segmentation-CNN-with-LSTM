from .convLSTM import *
from .MLLSTM import MLConvLSTM

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

class m2oBiConvLSTM(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])


    def __init__(self, input_channel, num_filter, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
        super(m2oBiConvLSTM, self).__init__()

        self.forward_net = ConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, padding=padding)
        self.backward_net = ConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        # self.bn = nn.BatchNorm2d(output_channel)
        self.bn = nn.BatchNorm3d(output_channel)

        self.m_length = m_length

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):

        # print(forward.shape)

        frt = forward[:, :self.m_length, :, :, :]    # b, 0~m_length, c, h, w
        bck = forward[:, -self.m_length: , :, :, :]  # b, m_length~-1, c, h ,w
        bck = self._make_reverse(bck)

        # print(torch.equal(frt[0][-1], bck[0][-1]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        # many to one
        # print(torch.equal(frt[:,-1, ...].squeeze(), frt[:, -1:, ... ].squeeze()))
        out_fwd = out_fwd[:,-1:,...]
        out_bwd = out_bwd[:,-1:,...]

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)
        ycat, t = self.voltoslices(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)
        # print(ycat.shape)
        # print("======")
        ## BN is necessary?

        return ycat, (f_state, b_state)

# class m2oMLBiConvLSTM(nn.Module):
#     @staticmethod
#     def _make_reverse(input):
#         return torch.flip(input, dims=[1])
#
#     @staticmethod
#     def voltoslices(input):
#         b, t, c, h, w = input.shape
#         return input.view(-1, c, h, w), t
#
#     @staticmethod
#     def slicetovol(input, t):
#         return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])
#
#     def __init__(self, input_channel, num_filter, num_layer, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
#         super(m2oMLBiConvLSTM, self).__init__()
#
#         self.forward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
#                                     kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
#         self.backward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
#                                      kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
#         self.conv11 = nn.Conv2d(in_channels=num_filter,
#                                 out_channels=output_channel,
#                                 kernel_size=(1, 1),
#                                 stride=1)
#
#         # self.bn = nn.BatchNorm2d(output_channel)
#         # self.bn = nn.BatchNorm3d(output_channel)
#         # self.inn = nn.InstanceNorm3d(output_channel)
#         self.inn = nn.InstanceNorm2d(output_channel)
#
#         self.m_length = m_length
#
#     # inputs and states should not be all none
#     # inputs: S*B*C*H*W => B*S*C*H*W
#
#     def forward(self, forward, device='cuda'):
#         # print(forward.shape)
#
#         frt = forward[:, :self.m_length, :, :, :]  # b, 0~m_length, c, h, w
#         bck = forward[:, -self.m_length:, :, :, :]  # b, m_length~-1, c, h ,w
#         bck = self._make_reverse(bck)
#
#         # print(torch.equal(frt[0][-1], bck[0][-1]))
#         # print(frt.shape)
#
#         # print(id(forward), id(backward))
#         # print(torch.equal(forward[0][3], backward[0][-4]))
#
#         out_fwd, f_state = self.forward_net(frt, device=device)
#         out_bwd, b_state = self.backward_net(bck, device=device)
#
#         # print(out_fwd.shape)
#         # print(out_bwd.shape)
#
#         # many to one
#         # print(torch.equal(frt[:,-1, ...].squeeze(), frt[:, -1:, ... ].squeeze()))
#         out_fwd = out_fwd[:, -1:, ...]
#         out_bwd = out_bwd[:, -1:, ...]
#
#         # print(out_fwd.shape)
#         # print(out_bwd.shape)
#
#         ycat = torch.cat((out_fwd, out_bwd), dim=2)
#         ycat, t = self.voltoslices(ycat)
#         ycat = self.conv11(ycat)
#         ycat = self.inn(ycat)
#         ycat = self.slicetovol(ycat, t)
#
#
#         # 3D INN
#         # ycat = ycat.permute(0,2,1,3,4) # B C T H W
#         # ycat = self.inn(ycat)
#         # ycat = ycat.permute(0, 2, 1, 3, 4) # B T C H W
#
#
#         # print(ycat.shape)
#         # print("======")
#         ## BN is necessary?
#
#         return ycat, (f_state, b_state)

class m2oMLConvLSTM(nn.Module):
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, num_layer, stride=1, padding=1, only_last=True):
        super().__init__()

        kernel_size = self._extend_for_multilayer(kernel_size, num_layer)
        num_filter = self._extend_for_multilayer(num_filter, num_layer)
        num_filter[-1] = 1
        stride = self._extend_for_multilayer(stride, num_layer)
        padding = self._extend_for_multilayer(padding, num_layer)
        self.num_layer = num_layer
        self.only_last = only_last

        if not len(kernel_size) == len(num_filter) == num_layer:
            raise ValueError("Not Match layer param length")

        layer_list = []

        for i in range(0, num_layer):
            cur_input_channel = input_channel if i == 0 else num_filter[i - 1]

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

        ## many to one
        layer_output_list=layer_output_list[:,-1:,...]
        # print(layer_output_list.shape)
        return layer_output_list, layer_state_list


class m2oNetBack(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2oBiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)




        return output, state





class m2oNetBi3DBack(nn.Module):


    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=conv_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(conv_channel),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels=conv_channel, out_channels=conv_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(conv_channel),
            nn.ReLU(),
        )
        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = BiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3))

        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        ## B S C H W -> B C S H W
        x = x.permute(0, 2 ,1, 3, 4)

        output = self.conv_block1(x)
        output = self.conv_block2(output)

        output = output.permute(0,2,1, 3, 4) # B T C H W

        # print(output.shape)
        output, state = self.lstm(output, device=device)
        # print(output.shape)
        output = output[:,self.m_length-1,:,:]
        # print(output.shape)

        return output, state


############# LAST MISSIOn
class m2o3DBNNetBack(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2o3DBNBiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)




        return output, state

class m2o3DBNBiConvLSTM(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])


    def __init__(self, input_channel, num_filter, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
        super(m2o3DBNBiConvLSTM, self).__init__()

        self.forward_net = ConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, padding=padding)
        self.backward_net = ConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        self.bn = nn.BatchNorm3d(num_filter)

        self.m_length = m_length

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):

        # print(forward.shape)

        frt = forward[:, :self.m_length, :, :, :]    # b, 0~m_length, c, h, w
        bck = forward[:, -self.m_length: , :, :, :]  # b, m_length~-1, c, h ,w
        bck = self._make_reverse(bck)

        # print(torch.equal(frt[0][-1], bck[0][-1]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        # many to one
        # print(torch.equal(frt[:,-1, ...].squeeze(), frt[:, -1:, ... ].squeeze()))
        out_fwd = out_fwd[:,-1:,...]
        out_bwd = out_bwd[:,-1:,...]

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)

        ycat = ycat.permute(1,2,0,3,4) # B C T H W
        ycat = self.bn(ycat)
        ycat = ycat.permute(0, 2, 1, 3, 4) # B T C H W

        ycat, t = self.voltoslices(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)

        return ycat, (f_state, b_state)

###
class m2oNetBiBack(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = BiConv3DBNLSTM(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3))

        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)
        # print(output.shape)
        output, state = self.lstm(output, device=device)
        # print(output.shape)
        output = output[:,self.m_length-1,:,:]
        # print(output.shape)

        return output, state

class BiConv3DBNLSTM(nn.Module):

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, input_channel, num_filter, output_channel, b_h_w, kernel_size, stride=1, padding=1):
        super(BiConv3DBNLSTM, self).__init__()

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
        self.bn = nn.BatchNorm3d(num_filter)

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

        ycat = ycat.permute(0,2,1,3,4) # B C T H W
        # print(ycat.shape)
        ycat = self.bn(ycat)
        ycat = ycat.permute(0, 2, 1, 3, 4) # B T C H W


        # print(ycat.shape)
        b, t, c, h, w = ycat.shape
        ycat = ycat.reshape(-1, c, h ,w) # B*T C H W
        # print(ycat.shape)

        ycat = self.conv11(ycat)
        ycat = ycat.reshape(-1, t, ycat.shape[1], h, w)
        # print(ycat.shape)

        # print(ycat.shape)

        ## BN is necessary?

        return ycat, (f_state)
###
class m2oNetMLBiBack(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, num_layer, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2oMLBiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,num_layer=num_layer)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)




        return output, state

class m2oMLBiConvLSTM(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, input_channel, num_filter, num_layer, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
        super(m2oMLBiConvLSTM, self).__init__()

        self.forward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.backward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        self.bn = nn.BatchNorm3d(num_filter)

        self.m_length = m_length

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):
        # print(forward.shape)

        frt = forward[:, :self.m_length, :, :, :]  # b, 0~m_length, c, h, w
        bck = forward[:, -self.m_length:, :, :, :]  # b, m_length~-1, c, h ,w
        bck = self._make_reverse(bck)

        # print(torch.equal(frt[0][-1], bck[0][-1]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        # many to one
        # print(torch.equal(frt[:,-1, ...].squeeze(), frt[:, -1:, ... ].squeeze()))
        out_fwd = out_fwd[:, -1:, ...]
        out_bwd = out_bwd[:, -1:, ...]

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)

        ycat = ycat.permute(1,2,0,3,4) # B C T H W
        ycat = self.bn(ycat)
        ycat = ycat.permute(0, 2, 1, 3, 4) # B T C H W

        ycat, t = self.voltoslices(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)



        return ycat, (f_state, b_state)
###
###
###
class m2o2DBNNetBack(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2o2DBNBiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)




        return output, state

class m2o2DBNBiConvLSTM(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])


    def __init__(self, input_channel, num_filter, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
        super(m2o2DBNBiConvLSTM, self).__init__()

        self.forward_net = ConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, padding=padding)
        self.backward_net = ConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        self.bn = nn.BatchNorm2d(num_filter)

        self.m_length = m_length

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):

        # print(forward.shape)

        frt = forward[:, :self.m_length, :, :, :]    # b, 0~m_length, c, h, w
        bck = forward[:, -self.m_length: , :, :, :]  # b, m_length~-1, c, h ,w
        bck = self._make_reverse(bck)

        # print(torch.equal(frt[0][-1], bck[0][-1]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        # many to one
        # print(torch.equal(frt[:,-1, ...].squeeze(), frt[:, -1:, ... ].squeeze()))
        out_fwd = out_fwd[:,-1:,...]
        out_bwd = out_bwd[:,-1:,...]

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)

        ycat, t = self.voltoslices(ycat)
        ycat = self.bn(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)

        return ycat, (f_state, b_state)
###
class m2oNetBiBack_2DBN(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = BiConv2DBNLSTM(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3))

        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)
        # print(output.shape)
        output, state = self.lstm(output, device=device)
        # print(output.shape)
        output = output[:,self.m_length-1,:,:]
        # print(output.shape)

        return output, state

class BiConv2DBNLSTM(nn.Module):

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, input_channel, num_filter, output_channel, b_h_w, kernel_size, stride=1, padding=1):
        super(BiConv2DBNLSTM, self).__init__()

        self.forward_net = ConvLSTM(input_channel=input_channel, num_filter= num_filter//2, b_h_w=b_h_w,
                                        kernel_size=kernel_size, stride=stride, padding=padding)
        self.backward_net = ConvLSTM(input_channel=input_channel, num_filter= num_filter//2, b_h_w=b_h_w,
                                        kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        # biconvlstm NetBack
        self.bn = nn.BatchNorm2d(num_filter)

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
        ycat = ycat.reshape(-1, c, h ,w) # B*T C H W
        # print(ycat.shape)
        ycat = self.bn(ycat)
        ycat = self.conv11(ycat)
        ycat = ycat.reshape(-1, t, ycat.shape[1], h, w)
        # print(ycat.shape)

        # print(ycat.shape)

        ## BN is necessary?

        return ycat, (f_state)
###
###
###
class m2oNetMLBiBack_2DBN(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, num_layer, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2oMLBiConvLSTM_2DBN(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,num_layer=num_layer)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)




        return output, state

class m2oMLBiConvLSTM_2DBN(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, input_channel, num_filter, num_layer, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
        super(m2oMLBiConvLSTM_2DBN, self).__init__()

        self.forward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.backward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        self.bn = nn.BatchNorm2d(num_filter)

        self.m_length = m_length

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):
        # print(forward.shape)

        frt = forward[:, :self.m_length, :, :, :]  # b, 0~m_length, c, h, w
        bck = forward[:, -self.m_length:, :, :, :]  # b, m_length~-1, c, h ,w
        bck = self._make_reverse(bck)

        # print(torch.equal(frt[0][-1], bck[0][-1]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        # many to one
        # print(torch.equal(frt[:,-1, ...].squeeze(), frt[:, -1:, ... ].squeeze()))
        out_fwd = out_fwd[:, -1:, ...]
        out_bwd = out_bwd[:, -1:, ...]

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)

        ycat, t = self.voltoslices(ycat)
        ycat = self.bn(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)



        return ycat, (f_state, b_state)
###
class m2oNet3Back(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)
        self.conv_block3 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2oBiConvLSTM(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)
        output = self.conv_block3(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)




        return output, state
###
###
###
class m2o1NetMLBiBack_2DBN(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, num_layer, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2oMLBiConvLSTM_2DBN(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,num_layer=num_layer)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)

        return output, state

###
class m2o1NetMLBiBack_2DBN_fuse(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, num_layer, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.mlstm = m2oMLBiConvLSTM_mid(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=lstm_channel,
                                  kernel_size=(3,3), m_length=m_length,num_layer=num_layer-1)
        self.olstm = m2oBiConvLSTM(input_channel=lstm_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                   kernel_size=(3,3), m_length=m_length)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)

        output = self.slicetovol(output, t)

        output, state = self.mlstm(output, device=device)
        output, state = self.olstm(output, device=device)

        return output, state

class m2oMLBiConvLSTM_mid(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, input_channel, num_filter, num_layer, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
        super(m2oMLBiConvLSTM_mid, self).__init__()

        self.forward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.backward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        self.bn = nn.BatchNorm2d(num_filter)

        self.m_length = m_length

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):
        # print(forward.shape)

        frt = forward[:, :self.m_length, :, :, :]  # b, 0~m_length, c, h, w
        bck = forward[:, -self.m_length:, :, :, :]  # b, m_length~-1, c, h ,w
        bck = self._make_reverse(bck)

        # print(torch.equal(frt[0][-1], bck[0][-1]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)

        ycat, t = self.voltoslices(ycat)
        ycat = self.bn(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)



        return ycat, (f_state, b_state)
###
class m2o2NetMLBiBack_2DBN_fuse(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, num_layer, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.mlstm = m2oMLBiConvLSTM_mid(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=lstm_channel,
                                  kernel_size=(3,3), m_length=m_length,num_layer=num_layer-1)
        self.olstm = m2oBiConvLSTM(input_channel=lstm_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                   kernel_size=(3,3), m_length=m_length)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.mlstm(output, device=device)
        output, state = self.olstm(output, device=device)

        return output, state

class m2o1NetMLBiBack_woBN(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, num_layer, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2oMLBiConvLSTM_woBN(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,num_layer=num_layer)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)

        return output, state

class m2oMLBiConvLSTM_woBN(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, input_channel, num_filter, num_layer, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
        super(m2oMLBiConvLSTM_woBN, self).__init__()

        self.forward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.backward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        # self.bn = nn.BatchNorm3d(num_filter)

        self.m_length = m_length

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):
        # print(forward.shape)

        frt = forward[:, :self.m_length, :, :, :]  # b, 0~m_length, c, h, w
        bck = forward[:, -self.m_length:, :, :, :]  # b, m_length~-1, c, h ,w
        bck = self._make_reverse(bck)

        # print(torch.equal(frt[0][-1], bck[0][-1]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        # many to one
        # print(torch.equal(frt[:,-1, ...].squeeze(), frt[:, -1:, ... ].squeeze()))
        out_fwd = out_fwd[:, -1:, ...]
        out_bwd = out_bwd[:, -1:, ...]

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)

        ycat, t = self.voltoslices(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)



        return ycat, (f_state, b_state)



############
class m2o2NetMLBiBack_custom_2DBN_fuse(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel1, lstm_channel2, m_length, num_layer, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.mlstm1 = m2oMLBiConvLSTM_inter(input_channel=conv_channel, num_filter=conv_channel, b_h_w=b_h_w, output_channel=conv_channel,
                                  kernel_size=(3,3),num_layer=1)
        self.mlstm2 = m2oMLBiConvLSTM_inter(input_channel=conv_channel, num_filter=lstm_channel1, b_h_w=b_h_w, output_channel=lstm_channel2,
                                  kernel_size=(3,3),num_layer=num_layer-2)
        self.olstm = m2oBiConvLSTM(input_channel=lstm_channel2, num_filter=lstm_channel2, b_h_w=b_h_w, output_channel=1,
                                   kernel_size=(3,3), m_length=m_length)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)

        output = self.slicetovol(output, t)
        # print("before",output.shape)
        output, state = self.mlstm1(output, device=device)
        # print(output.shape)
        output, state = self.mlstm2(output, device=device)
        # print(output.shape)
        output, state = self.olstm(output, device=device)

        return output, state

class m2oMLBiConvLSTM_inter(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, input_channel, num_filter, num_layer, output_channel, b_h_w, kernel_size, stride=1, padding=1):
        super(m2oMLBiConvLSTM_inter, self).__init__()

        self.forward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.backward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        self.bn = nn.BatchNorm2d(num_filter)


    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):
        # print(forward.shape)

        frt = forward
        bck = self._make_reverse(frt)

        # print(torch.equal(frt[0][-1], bck[0][0]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)

        ycat, t = self.voltoslices(ycat)
        ycat = self.bn(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)



        return ycat, (f_state, b_state)
##
class m2o2NetMLBiBack_3DBN(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, num_layer, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2oMLBiConvLSTM_3DBN(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,num_layer=num_layer)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)

        return output, state

class m2oMLBiConvLSTM_3DBN(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, input_channel, num_filter, num_layer, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
        super(m2oMLBiConvLSTM_3DBN, self).__init__()

        self.forward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.backward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        self.bn = nn.BatchNorm3d(num_filter)

        self.m_length = m_length

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):
        # print(forward.shape)

        # frt = forward[:, :self.m_length, :, :, :]  # b, 0~m_length, c, h, w
        # bck = forward[:, -self.m_length:, :, :, :]  # b, m_length~-1, c, h ,w
        frt = forward
        bck = frt
        bck = self._make_reverse(bck)

        # print(torch.equal(frt[0][-1], bck[0][-1]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        # many to one
        # print(torch.equal(frt[:,-1, ...].squeeze(), frt[:, -1:, ... ].squeeze()))
        out_fwd = out_fwd[:, [self.m_length], ...]
        out_bwd = out_bwd[:, [self.m_length], ...]

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)
        # print(ycat.shape)

        ycat = ycat.permute(0,2,1,3,4) # B C T H W
        ycat = self.bn(ycat)
        ycat = ycat.permute(0,2,1,3,4) # B T C H W

        ycat, t = self.voltoslices(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)



        return ycat, (f_state, b_state)
###
class m2oNet1MLBi_2DBN(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, num_layer, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        # self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2oMLBi_2DBN(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,num_layer=num_layer)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        # output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)




        return output, state

class m2oMLBi_2DBN(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, input_channel, num_filter, num_layer, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
        super(m2oMLBi_2DBN, self).__init__()

        self.forward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.backward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        self.bn = nn.BatchNorm2d(num_filter)

        self.m_length = m_length

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):
        # print(forward.shape)

        # frt = forward[:, :self.m_length, :, :, :]  # b, 0~m_length, c, h, w
        # bck = forward[:, -self.m_length:, :, :, :]  # b, m_length~-1, c, h ,w
        frt = forward
        bck = frt
        bck = self._make_reverse(bck)

        # print(torch.equal(frt[0][-1], bck[0][-1]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        # many to one
        # print(torch.equal(frt[:,-1, ...].squeeze(), frt[:, -1:, ... ].squeeze()))
        # print(out_fwd.shape)
        # print(out_bwd.shape)

        out_fwd = out_fwd[:, [self.m_length-1], ...]
        out_bwd = out_bwd[:, [self.m_length-1], ...]



        # print(out_fwd.shape)
        # print(out_bwd.shape)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)

        ycat, t = self.voltoslices(ycat)
        ycat = self.bn(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)



        return ycat, (f_state, b_state)
###
class m2oNet1MLBi(nn.Module):
    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    def __init__(self, in_channels, conv_channel, lstm_channel, m_length, num_layer, HALF_MODE=False):
        super().__init__()

        self.conv_block1 = straightConv(in_channels=in_channels, conv_channel=conv_channel)
        # self.conv_block2 = straightConv(in_channels=conv_channel, conv_channel=conv_channel)

        if HALF_MODE:
            b_h_w = (1, 256, 256)
        else:
            b_h_w = (1, 512, 512)

        self.m_length = m_length

        self.lstm = m2oMLBi(input_channel=conv_channel, num_filter=lstm_channel, b_h_w=b_h_w, output_channel=1,
                                  kernel_size=(3,3), m_length=m_length,num_layer=num_layer)
        # self.conv11 = nn.Conv2d(in_channels=lstm_channel, out_channels=1, kernel_size=(1,1),stride=1)

    def forward(self, x, device):

        output, t = self.voltoslices(x)

        output = self.conv_block1(output)
        # output = self.conv_block2(output)

        output = self.slicetovol(output, t)

        output, state = self.lstm(output, device=device)




        return output, state

class m2oMLBi(nn.Module):
    @staticmethod
    def _make_reverse(input):
        return torch.flip(input, dims=[1])

    @staticmethod
    def voltoslices(input):
        b, t, c, h, w = input.shape
        return input.view(-1, c, h, w), t

    @staticmethod
    def slicetovol(input, t):
        return input.view(-1, t, input.shape[1], input.shape[2], input.shape[3])

    def __init__(self, input_channel, num_filter, num_layer, output_channel, b_h_w, kernel_size, m_length, stride=1, padding=1):
        super(m2oMLBi, self).__init__()

        self.forward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.backward_net = MLConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, num_layer=num_layer, padding=padding, only_last=True)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        # self.bn = nn.BatchNorm2d(num_filter)

        self.m_length = m_length

    # inputs and states should not be all none
    # inputs: S*B*C*H*W => B*S*C*H*W

    def forward(self, forward, device='cuda'):
        # print(forward.shape)

        # frt = forward[:, :self.m_length, :, :, :]  # b, 0~m_length, c, h, w
        # bck = forward[:, -self.m_length:, :, :, :]  # b, m_length~-1, c, h ,w
        frt = forward
        bck = frt
        bck = self._make_reverse(bck)

        # print(torch.equal(frt[0][-1], bck[0][-1]))
        # print(frt.shape)

        # print(id(forward), id(backward))
        # print(torch.equal(forward[0][3], backward[0][-4]))

        out_fwd, f_state = self.forward_net(frt, device=device)
        out_bwd, b_state = self.backward_net(bck, device=device)

        # print(out_fwd.shape)
        # print(out_bwd.shape)

        # many to one
        # print(torch.equal(frt[:,-1, ...].squeeze(), frt[:, -1:, ... ].squeeze()))
        # print(out_fwd.shape)
        # print(out_bwd.shape)

        out_fwd = out_fwd[:, [self.m_length-1], ...]
        out_bwd = out_bwd[:, [self.m_length-1], ...]



        # print(out_fwd.shape)
        # print(out_bwd.shape)

        ycat = torch.cat((out_fwd, out_bwd), dim=2)

        ycat, t = self.voltoslices(ycat)
        # ycat = self.bn(ycat)
        ycat = self.conv11(ycat)
        ycat = self.slicetovol(ycat, t)



        return ycat, (f_state, b_state)