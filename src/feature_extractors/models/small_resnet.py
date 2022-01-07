from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, c, downsample: bool = False) -> None:
        super(ResidualBlock, self).__init__()
        layers = [
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU()
        ]
        if downsample:
            layers += [
                nn.Conv2d(c, c * 2, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(c * 2),
            ]
            self.downsampling_conv = nn.Conv2d(c, c * 2, kernel_size=3, padding=1, stride=2)
        else:
            layers += [
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
            ]
        self.block = nn.Sequential(*layers)
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.downsample:
            return self.relu(self.block(x) + self.downsampling_conv(x))
        return self.relu(self.block(x) + x)


class SmallResnet(nn.Module):
    def __init__(self, num_layers: int, num_classes: int = 10, num_channels: int = 16, num_input_channels: int = 3) -> None:
        super(SmallResnet, self).__init__()
        if num_layers%6 != 2:
            raise AssertionError("num_layers has to equal 2 mod 6")
        num_blocks = (num_layers-2)//6
        num_channels = [num_channels, num_channels * 2, num_channels * 4]
        layers = [
            nn.Conv2d(num_input_channels, num_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU()
        ]
        for i, c in enumerate(num_channels):
            for j in range(num_blocks):
                if j == num_blocks - 1 and i != len(num_channels) - 1:
                    layers.append(ResidualBlock(c, downsample=True))
                else:
                    layers.append(ResidualBlock(c))

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(num_channels[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        feature_extractor.eval()
        return feature_extractor(x)
