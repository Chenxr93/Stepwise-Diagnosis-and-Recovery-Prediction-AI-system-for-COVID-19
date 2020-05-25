import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=[1., 1.], gamma=2.):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        if input.min() < 0:
            input = torch.sigmoid(input)
        input = input.squeeze(1)

        bce_loss = F.binary_cross_entropy(input, target.float(), reduction='none')
        neg_loss = self.alpha[0] * torch.pow((input), self.gamma) * (1 - target.float()) * bce_loss
        pos_loss = self.alpha[1] * torch.pow((1 - input), self.gamma) * target.float() * bce_loss

        loss = pos_loss + neg_loss

        return loss.mean()
