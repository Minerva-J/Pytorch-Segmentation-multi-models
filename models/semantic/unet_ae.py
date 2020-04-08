# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from models import backbone
from utils_Deeplab import SyncBN2d
from modules import SCSEBlock


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = SyncBN2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = SyncBN2d(out_channels)
        self.downsample = downsample
        if (self.downsample is None) and (in_channels != out_channels):
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                            SyncBN2d(out_channels))
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = SyncBN2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = SyncBN2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = SyncBN2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if (self.downsample is None) and (in_channels != out_channels):
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                            SyncBN2d(out_channels))
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Decoder(nn.Module):

    def __init__(self, block, in_channels, channels, out_channels, reduction=16, stride=1,
                 downsample=None):
        super(Decoder, self).__init__()
        self.block1 = block(in_channels, channels, stride, downsample)
        self.block2 = block(channels, out_channels, stride, downsample)
        self.scse_module = SCSEBlock(out_channels, reduction=reduction)

    def forward(self, x, e=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.scse_module(x)
        return x


class UnetResnetAE(nn.Module):
    def __init__(self, in_channels, num_classes, backend='resnet18', pretrained='imagenet'):
        super(UnetResnetAE, self).__init__()
        self.in_channes = in_channels
        self.num_classes = num_classes
        if 'resne' in backend:
            self.encoder = ResnetBackend(backend, pretrained)
        else:
            raise NotImplementedError

        if backend in ['resnet18', 'resnet34']:
            block = BasicBlock
        else:
            block = Bottleneck

        inter_channel = self.encoder.lastconv_channel

        self.center = nn.Sequential(nn.Conv2d(inter_channel, 512,
                                              kernel_size=3, padding=1, bias=False),
                                    SyncBN2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                                    SyncBN2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.decoder5 = Decoder(block, 256 + inter_channel, 512, 64)
        self.decoder4 = Decoder(block, 64 + inter_channel // 2, 256, 64)
        self.decoder3 = Decoder(block, 64 + inter_channel // 4, 128, 64)
        self.decoder2 = Decoder(block, 64 + inter_channel // 8, 64, 64)
        self.decoder1 = Decoder(block, 64, 32, 64)

        self.cbr_last = nn.Sequential(nn.Conv2d(64 * 6, 64, kernel_size=3, padding=1, bias=False),
                                      SyncBN2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, self.num_classes, kernel_size=1))

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        d = self.center(x4)
        d5 = self.decoder5(d, x4)
        d4 = self.decoder4(d5, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2)

        out = torch.cat([F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=True),
                         d1,
                         F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True),
                         F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True),
                         F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True),
                         F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=True)], 1)
        out = F.dropout2d(out, .5)
        out = self.cbr_last(out)
        return out


class ResnetBackend(nn.Module):
    def __init__(self, backend='resnet18', pretrained='imagenet'):
        '''
        :param backend: resnet<> or se_resnet
        '''
        super(ResnetBackend, self).__init__()
        _all_resnet_models = backbone._all_resnet_backbones
        if backend not in _all_resnet_models:
            raise Exception(f"{backend} must in {_all_resnet_models}")

        _backend_model = backbone.__dict__[backend](pretrained=pretrained)
        if 'se' in backend:
            self.feature0 = nn.Sequential(_backend_model.layer0.conv1,
                                          _backend_model.layer0.bn1,
                                          _backend_model.layer0.relu1)
        else:
            self.feature0 = nn.Sequential(_backend_model.conv1,
                                          _backend_model.bn1,
                                          _backend_model.relu)
        self.feature1 = _backend_model.layer1
        self.feature2 = _backend_model.layer2
        self.feature3 = _backend_model.layer3
        self.feature4 = _backend_model.layer4

        if backend in ['resnet18', 'resnet34']:
            self.lastconv_channel = 512
        else:
            self.lastconv_channel = 512 * 4

    def forward(self, x):
        x0 = self.feature0(x)  # 1/2
        x1 = self.feature1(x0)  # 1/2
        x2 = self.feature2(x1)  # 1/4
        x3 = self.feature3(x2)  # 1/8
        x4 = self.feature4(x3)  # 1/16
        return x0, x1, x2, x3, x4


if __name__ == '__main__':
    from torchsummary import summary

    unet_resnet = UnetResnetAE(in_channels=3, num_classes=21, backend='resnet101').cuda()
    x = torch.randn(2, 3, 224, 224)
    out = unet_resnet(x.cuda())
    print(summary(unet_resnet, [3, 224, 224]))
    print('Total params: ', sum(p.numel() for p in unet_resnet.parameters() if p.requires_grad))
    print(out.size())
