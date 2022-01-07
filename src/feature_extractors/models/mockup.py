from torch import nn
import torch

class Mockup(nn.Module):
    def __init__(self, num_classes):
        super(Mockup, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, **kwargs):
        return torch.randn((x.shape[0], 512)), torch.randn((x.shape[0], self.num_classes))
