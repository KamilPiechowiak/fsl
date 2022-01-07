from torch import nn
import numpy as np

from .layers import Mixup, NormLinear

class ResidualBlock(nn.Module):
    def __init__(self, c, downsample: bool = False) -> None:
        super(ResidualBlock, self).__init__()
        layers = []
        if downsample:
            layers += [
                nn.Conv2d(c // 2, c, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(c),
                nn.ReLU()
            ]
            self.downsampling_conv = nn.Conv2d(c // 2, c, kernel_size=1, padding=0, stride=2)
        else:
            layers += [
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU()
            ]
        layers += [
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c)
        ]
        self.block = nn.Sequential(*layers)
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.downsample:
            return self.relu(self.block(x) + self.downsampling_conv(x))
        return self.relu(self.block(x) + x)


class Resnet(nn.Module):
    def __init__(self, num_layers: int, num_classes: int = 10, num_channels: int = 16, num_input_channels: int = 3) -> None:
        super(Resnet, self).__init__()
        if num_layers%8 != 2:
            raise AssertionError("num_layers has to equal 2 mod 8")
        num_blocks = (num_layers-2)//8
        num_channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
        self.num_layers = len(num_channels)+1
        setattr(self, "layer0", nn.Sequential(
            nn.Conv2d(num_input_channels, num_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU()
        ))
        for i, c in enumerate(num_channels):
            block = []
            for j in range(num_blocks):
                if j == 0 and i != 0:
                    block.append(ResidualBlock(c, downsample=True))
                else:
                    block.append(ResidualBlock(c))
            setattr(self, f"layer{i+1}", nn.Sequential(*block))
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
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i}")
            x = layer(x)
            if i+1 == mixup_level:
                x, labels1, labels2 = self.mixup(x, labels, lambda_)
        x = self.pool(x)
        x = self.flatten(x)
        features = x
        x = self.linear(x)
        if labels is not None:
            return features, x, labels1, labels2
        return features, x

    def extract_features(self, x):
        return self.forward(x)[0]

