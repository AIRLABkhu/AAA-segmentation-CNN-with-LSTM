from torch import nn
import torch
from models.convLSTM import *

def set_grad(model, grad):
    for param in model.parameters():
        param.requires_grad = grad

class DBiConvLSTM(nn.Module):
    def __init__(self, input_channel, num_filter, output_channel, b_h_w, kernel_size, stride=1, padding=1,
                 device='cuda'):
        super(DBiConvLSTM, self).__init__()

        self.forward_net = ConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                    kernel_size=kernel_size, stride=stride, padding=padding, device=device)
        self.backward_net = ConvLSTM(input_channel=input_channel, num_filter=num_filter // 2, b_h_w=b_h_w,
                                     kernel_size=kernel_size, stride=stride, padding=padding, device=device)
        self.conv11 = nn.Conv2d(in_channels=num_filter,
                                out_channels=output_channel,
                                kernel_size=(1, 1),
                                stride=1)

        self.bn = nn.BatchNorm2d(output_channel)

        # inputs and states should not be all none
        # inputs: S*B*C*H*W => B*S*C*H*W

    def set_freeze(self, fwd=False, bwd=False):
        set_grad(self.forward_net, not fwd)
        set_grad(self.backward_net, not bwd)

    def forward(self, forward, backward, device='cuda'):
        out_fwd, _ = self.forward_net(forward, device=device)
        out_bwd, _ = self.backward_net(backward, device=device)
        # t b c h w

        reversed_idx = list(reversed(range(out_bwd.shape[0])))
        out_bwd = out_bwd[reversed_idx, ...]  # reverse temporal outputs.

        # print("=======")

        ycat = torch.cat((out_fwd, out_bwd), dim=2)
        # print(ycat.shape)
        t, b, c, h, w = ycat.shape
        ycat = ycat.view(-1, c, h, w)  # B*T C H W
        # print(ycat.shape)

        ycat = self.conv11(ycat)
        ycat = self.bn(ycat)
        # print(ycat.shape)
        ycat = ycat.view(t, -1, ycat.shape[1], h, w)
        # print(ycat.shape)

        ## BN is necessary?

        return ycat