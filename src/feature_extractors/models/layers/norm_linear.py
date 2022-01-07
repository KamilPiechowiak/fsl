import torch
from torch import nn

class NormLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(NormLinear, self).__init__()
        self.linear = torch.nn.utils.weight_norm(nn.Linear(in_features, out_features))
    
    def forward(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return self.linear(x)