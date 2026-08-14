"""U-Net architecture used to create models/best_model.pt."""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 13, out_channels: int = 1, features=(32, 64, 128, 256)):
        super().__init__()
        self.down_blocks, self.up_transpose, self.up_blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        current_channels = in_channels
        for feature in features:
            self.down_blocks.append(DoubleConv(current_channels, feature))
            current_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        current_channels = features[-1] * 2
        for feature in reversed(features):
            self.up_transpose.append(nn.ConvTranspose2d(current_channels, feature, 2, 2))
            self.up_blocks.append(DoubleConv(feature * 2, feature))
            current_channels = feature
        self.output = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for block in self.down_blocks:
            x = block(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for up, block, skip in zip(self.up_transpose, self.up_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((skip, x), dim=1))
        return self.output(x)
