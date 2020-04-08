# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from modules import unetConv2d, unetDown, unetUp


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, filter_scale=1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.filter_scale = filter_scale
        filters = [64, 128, 256, 512, 1024]
        self.filters = [int(x / self.filter_scale) for x in filters]

        self.in_conv = unetConv2d(in_channels, self.filters[0])
        self.down, self.up = self._make_layers(self.filters)
        self.out_conv = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        n = len(self.filters)
        conv_outputs = []
        conv_outputs.append(self.in_conv(x))
        for i in range(n - 1):
            conv_outputs.append(self.down[i](conv_outputs[i]))

        out = conv_outputs[-1]
        for j in range(n - 1):
            out = self.up[j](out, conv_outputs[n - 2 - j])

        return self.out_conv(out)

    def _make_layers(self, filters):
        down, up = [], []
        n = len(filters)
        for i in range(n - 1):
            down.append(unetDown(filters[i], filters[i + int(i != n - 2)]))
            up.append(unetUp(filters[n - i - 1], filters[(n - i - 3) * int(n - i - 3 > 0)]))
        return nn.ModuleList(down), nn.ModuleList(up)


if __name__ == '__main__':
    from torchsummary import summary

    unet = UNet(in_channels=3, num_classes=21, filter_scale=2).cuda()
    print(summary(unet, [3, 224, 224]))
    print('Total params: ', sum(p.numel() for p in unet.parameters() if p.requires_grad))
    x = torch.randn(1, 3, 224, 224)
    out = unet(x.cuda())
    print(out.size())
