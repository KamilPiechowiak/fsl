from torch import nn
import numpy as np

from .layers import Mixup, NormLinear

class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, downsample=False) -> None:
        super(ResidualBlock, self).__init__()
        transform_residual = (c_in != c_out)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu1 = nn.ReLU()
        self.block = []
        if downsample or transform_residual:
            if not downsample:
                stride = 1
            else:
                stride = 2
            self.block+= [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=stride)]
            self.residual_conv = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=stride)
        else:
            self.block+= [nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)]
            self.residual_conv = None
        
        self.block+= [
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        if self.residual_conv is not None:
            x = self.relu1(self.bn1(x))
            return self.block(x) + self.residual_conv(x)
        return self.block(self.relu1(self.bn1(x))) + x


class WideResnet(nn.Module):
    def __init__(self, num_layers: int, num_classes: int = 10, num_channels: int = 160, num_input_channels: int = 3) -> None:
        super(WideResnet, self).__init__()
        if num_layers%6 != 2:
            raise AssertionError("num_layers has to equal 2 mod 6")
        num_blocks = (num_layers-2)//6
        num_channels = [16, num_channels, num_channels * 2, num_channels * 4]
        self.conv1 = nn.Conv2d(num_input_channels, num_channels[0], kernel_size=3, padding=1)
        self.num_layers = len(num_channels)-1
        for i in range(1, len(num_channels)):
            block = []
            for j in range(num_blocks):
                downsample = False
                if j == 0 and i != 1:
                    downsample = True
                if j == 0:
                    block.append(ResidualBlock(num_channels[i-1], num_channels[i], downsample=downsample))
                else:
                    block.append(ResidualBlock(num_channels[i], num_channels[i]))
            setattr(self, f"layer{i}", nn.Sequential(*block))
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = NormLinear(num_channels[-1], num_classes)
        self.mixup = Mixup()

    def forward(self, x, labels=None, lambda_=None):
        mixup_level = None
        if lambda_ is not None:
            mixup_level = np.random.randint(0, self.num_layers+1)
        if mixup_level == 0:
            x, labels1, labels2 = self.mixup(x, labels, lambda_)
        x = self.conv1(x)
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i+1}")
            x = layer(x)
            if i+1 == mixup_level:
                x, labels1, labels2 = self.mixup(x, labels, lambda_)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        features = x
        x = self.linear(x)
        if labels is not None:
            return features, x, labels1, labels2
        return features, x

    def extract_features(self, x):
        return self.forward(x)[0]
