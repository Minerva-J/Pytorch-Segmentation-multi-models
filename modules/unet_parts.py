# -*- coding: utf-8 -*-
import math

import torch
from torch import nn
import torch.nn.functional as F
from utils_Deeplab import SyncBN2d


class unetConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(unetConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            SyncBN2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            SyncBN2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class unetDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(unetDown, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            unetConv2d(in_channel, out_channel)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class unetUp(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        # in_channel should be 2x in_channel
        super(unetUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

        self.conv = unetConv2d(in_channel, out_channel)

    def forward(self, x_small, x_big):
        x_small = self.up(x_small)
        x_small = F.interpolate(x_small, size=x_big.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_big, x_small], dim=1)
        x = self.conv(x)
        return x
