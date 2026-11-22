"""Petit U-Net (entrée RGB + masque GrabCut = 4 canaux)."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetRefiner(nn.Module):
    def __init__(self, base_channels: int = 12):
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv(4, b)
        self.enc2 = DoubleConv(b, 2*b)
        self.enc3 = DoubleConv(2*b, 4*b)
        self.center = DoubleConv(4*b, 8*b)
        self.dec3 = DoubleConv(8*b + 4*b, 4*b)
        self.dec2 = DoubleConv(4*b + 2*b, 2*b)
        self.dec1 = DoubleConv(2*b + b, b)
        self.out = nn.Conv2d(b, 1, 1)

    @staticmethod
    def up(x, skip):
        return torch.cat((F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip), 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        d = self.center(F.max_pool2d(c, 2))
        u = self.dec3(self.up(d, c))
        u = self.dec2(self.up(u, b))
        u = self.dec1(self.up(u, a))
        return self.out(u)
