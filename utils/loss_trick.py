import torch.nn as nn
import torch

"""
this focal have two keypoints:
(1) when the focal loss is the class nn.Module。 when uses it should be *.to(device)*
(2) if use this loss, the order just like this, focal_loss(a, b)
"""


class FocalLoss(nn.Module):
    def __init__(self, alpha, gama):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gama = gama

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduce=None)(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha*((1-pt)**self.gama)*ce_loss
        return F_loss.mean()

