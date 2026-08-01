import torch
from torch import nn


class ResBlock(nn.Module):
    """Residual block with an identity or projection skip connection."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.layer2(self.layer1(x))
        out += residual
        return self.relu(out)


class ResNet(nn.Module):
    """ResNet built from a stem convolution followed by four residual stages."""

    def __init__(self, in_channels: int, out_channels: int, blocks: list, block=ResBlock):
        super().__init__()
        self.in_channels = 64

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.block1 = self._make_layer(block, 64, blocks[0])
        self.block2 = self._make_layer(block, 128, blocks[1], stride=2)
        self.block3 = self._make_layer(block, 256, blocks[2], stride=2)
        self.block4 = self._make_layer(block, 512, blocks[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_channels)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Generates a Sequential repeatative resblock connections
        Args:
            planes (int): Number of output channels for the block.
            blocks (int): Number of blocks to create.
            stride (int, optional): Stride for the first block. Defaults to 1.
            block (ResBlock, optional): Block type to use. Defaults to ResBlock.
        Returns:
            nn.Sequential: A sequential container of the created blocks.
        """
        layers = [block(self.in_channels, planes, stride=stride)]
        self.in_channels = planes * block.expansion     # to update the inchannels after every block consturction for the next block
        layers += [block(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gap(self.block4(self.block3(self.block2(self.block1(self.layer1(x))))))
        out = torch.flatten(out, start_dim=1)
        return self.fc(out)
