"""
Based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""


import torch
from torch import Tensor, nn


from typing import Optional, Type, Union


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            identity_downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity_downsample = identity_downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.identity_downsample:
            identity = self.identity_downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            identity_downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.identity_downsample = identity_downsample
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_downsample:
            identity = self.identity_downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            num_classes: int,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: list
    ) -> None:
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, layers[0], self.in_channels)
        self.layer2 = self._make_layer(block, layers[1], 32, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 64, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._init_params()

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            num_blocks: int,
            out_channels: int,
            stride: int = 1
    ) -> nn.Sequential:
        identity_downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                          out_channels=out_channels * block.expansion,
                          kernel_size=1,
                          stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = list()
        layers.append(
            block(self.in_channels, out_channels, stride, identity_downsample)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    tensor=m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def resnet20(num_classes) -> ResNet:
    return ResNet(num_classes, BasicBlock, [3, 3, 3])
