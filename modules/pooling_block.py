# -*- coding: utf-8 -*-
import math

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from utils_Deeplab import SyncBN2d


class PPBlock(nn.Module):
    def __init__(self, in_channel=2048, out_channel=256, pool_scales=(1, 2, 3, 6)):
        '''
        :param in_channel: default 2048 for resnet50 backend
        :param out_channel: default 256 for resnet50 backend
        :param pool_scales: default scales [1,2,3,6]
        '''
        super(PPBlock, self).__init__()
        self.pp_bra1 = nn.Sequential(
            OrderedDict([('pool1', nn.AdaptiveAvgPool2d(pool_scales[0])),
                         ('conv1', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn1', SyncBN2d(out_channel)),
                         ('relu1', nn.ReLU(inplace=True))])
        )
        self.pp_bra2 = nn.Sequential(
            OrderedDict([('pool2', nn.AdaptiveAvgPool2d(pool_scales[1])),
                         ('conv2', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn2', SyncBN2d(out_channel)),
                         ('relu2', nn.ReLU(inplace=True))])
        )
        self.pp_bra3 = nn.Sequential(
            OrderedDict([('pool3', nn.AdaptiveAvgPool2d(pool_scales[2])),
                         ('conv3', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn3', SyncBN2d(out_channel)),
                         ('relu3', nn.ReLU(inplace=True))])
        )
        self.pp_bra4 = nn.Sequential(
            OrderedDict([('pool4', nn.AdaptiveAvgPool2d(pool_scales[3])),
                         ('conv4', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn4', SyncBN2d(out_channel)),
                         ('relu4', nn.ReLU(inplace=True))])
        )

    def forward(self, x):
        upsample = self._make_upsample(x.size()[2:])
        return torch.cat([x, upsample(self.pp_bra1(x)),
                          upsample(self.pp_bra2(x)),
                          upsample(self.pp_bra3(x)),
                          upsample(self.pp_bra4(x))], dim=1)

    def _make_upsample(self, size):
        return nn.Upsample(size=size, mode='bilinear', align_corners=True)


class ASPPBlock(nn.Module):
    def __init__(self, in_channel=2048, out_channel=256, os=16):
        '''
        :param in_channel: default 2048 for resnet101
        :param out_channel: default 256 for resnet101
        :param os: 16 or 8
        '''
        super(ASPPBlock, self).__init__()
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.gave_pool = nn.Sequential(
            OrderedDict([('gavg', nn.AdaptiveAvgPool2d(rates[0])),
                         ('conv0_1', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn0_1', SyncBN2d(out_channel)),
                         ('relu0_1', nn.ReLU(inplace=True))])
        )
        self.conv1_1 = nn.Sequential(
            OrderedDict([('conv0_2', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn0_2', SyncBN2d(out_channel)),
                         ('relu0_2', nn.ReLU(inplace=True))])
        )
        self.aspp_bra1 = nn.Sequential(
            OrderedDict([('conv1_1', nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                               padding=rates[1], dilation=rates[1], bias=False)),
                         ('bn1_1', SyncBN2d(out_channel)),
                         ('relu1_1', nn.ReLU(inplace=True))])
        )
        self.aspp_bra2 = nn.Sequential(
            OrderedDict([('conv1_2', nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                               padding=rates[2], dilation=rates[2], bias=False)),
                         ('bn1_2', SyncBN2d(out_channel)),
                         ('relu1_2', nn.ReLU(inplace=True))])
        )
        self.aspp_bra3 = nn.Sequential(
            OrderedDict([('conv1_3', nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                               padding=rates[3], dilation=rates[3], bias=False)),
                         ('bn1_3', SyncBN2d(out_channel)),
                         ('relu1_3', nn.ReLU(inplace=True))])
        )
        self.aspp_catdown = nn.Sequential(
            OrderedDict([('conv_down', nn.Conv2d(5 * out_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn_down', SyncBN2d(out_channel)),
                         ('relu_down', nn.ReLU(inplace=True)),
                         ('drop_out', nn.Dropout(.1))])
        )

    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool(x), size[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, self.conv1_1(x),
                       self.aspp_bra1(x),
                       self.aspp_bra2(x),
                       self.aspp_bra3(x)], dim=1)
        x = self.aspp_catdown(x)
        return x
