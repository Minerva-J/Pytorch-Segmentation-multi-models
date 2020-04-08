import torch.nn.functional as F

from .unet_parts import *
import torch.nn as nn
import torch
class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet4, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc1 = outconv(256, n_classes)
        self.outc2 = outconv(128, n_classes)
        self.outc3 = outconv(64, n_classes)
        self.outc4 = outconv(64, n_classes)
        # self.SM = nn.Sigmoid()
        # self.SM = torch.sigmoid()
        # self.LS = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x1 = self.inc(x)#256
        x2 = self.down1(x1)#128
        x3 = self.down2(x2)#64
        x4 = self.down3(x3)#32
        x5 = self.down4(x4)#16
        x6 = self.up1(x5, x4)#32+32
        
        x7 = self.up2(x6, x3)
        
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        # x6 = self.outc1(x6)
        # x7 = self.outc2(x7)
        # x8 = self.outc3(x8)
        x9 = self.outc4(x9)
        # x6 = nn.functional.interpolate(x6, scale_factor=(8, 8), mode='bilinear', align_corners=True)
        # x7 = nn.functional.interpolate(x7, scale_factor=(4, 4), mode='bilinear', align_corners=True)
        # x8 = nn.functional.interpolate(x8, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        # x = x6 + x7 + x8 + x9
        x = x9
        # print()
        # x = torch.sigmoid(x)
        # x = self.LS(x)
        # print('x', x.shape)
        return x