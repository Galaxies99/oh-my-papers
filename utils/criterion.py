import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, res, target):
        return self.cross_entropy(res, target.long())
