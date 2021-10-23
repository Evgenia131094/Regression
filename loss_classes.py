import torch
import torch.nn as nn

class ContinuousLoss_L2(nn.Module):
    ''' Class to measure loss between continuous emotion dimension predictions and labels. Using l2 loss as base. '''

    def __init__(self, margin=1):
        super(ContinuousLoss_L2, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        labs = torch.abs(pred - target)
        loss = labs ** 2
        loss[(labs < self.margin)] = 0.0
        return loss.sum()


class ContinuousLoss_SL1(nn.Module):
    ''' Class to measure loss between continuous emotion dimension predictions and labels. Using smooth l1 loss as base. '''

    def __init__(self, margin=1):
        super(ContinuousLoss_SL1, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        labs = torch.abs(pred - target)
        loss = 0.5 * (labs ** 2)
        loss[(labs > self.margin)] = labs[(labs > self.margin)] - 0.5
        return loss.sum()
