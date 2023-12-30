import torch
import torch.nn as nn

from repnets.models._repvgg_modules import RepVGGBlock
from repnets.models.generic_repmodel import RepModel


class SimpleNet(RepModel):
    def __init__(self, num_classes):
        super().__init__()
        self.stage1 = RepVGGBlock(
            in_channels=3,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.stage2 = RepVGGBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
