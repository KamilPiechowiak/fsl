import torch
from torch import nn

class Mixup(nn.Module):
    def __init__(self):
        super(Mixup, self).__init__()
    
    def forward(self, x, labels, lambda_):
        ids = torch.randperm(len(labels))
        return (1-lambda_)*x + lambda_*x[ids], labels, labels[ids]