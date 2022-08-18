import torch

class binaryfocalloss(torch.nn.Module):
    def __init__(self, alpha=3, gamma=1.5, threshold=0.1, ignore_index=None, reduction='mean', **kwargs):
        super(binaryfocalloss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # self.smooth = 1e-6
        self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.threshold = threshold

        assert self.reduction in ['none','mean','sum']

    def forward(self, output, target):
        prob = prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0-self.smooth)
        # prob sigmoid - > 0~1 / clamp -> side cut

        valid_mask = None
        if self.ignore_index is not None: #skip
            valid_mask = (target != self.ignore_index).float()


        # pos_mask = (target == 1).float()
        # neg_mask = (target == 0).float()

        pos_mask = (target >=self.threshold).float()
        neg_mask = (target < self.threshold).float()

        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1-prob, self.gamma)).detach()
        pos_loss = -self.alpha * pos_weight * torch.log(prob)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -neg_weight * torch.log(1-prob)
        loss = pos_loss + neg_loss
        loss = loss.mean()

        return loss