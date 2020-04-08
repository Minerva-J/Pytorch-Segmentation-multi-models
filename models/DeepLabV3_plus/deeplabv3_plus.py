# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from models import backbone
from modules import ASPPBlock
from utils_Deeplab import SyncBN2d


class DeepLabv3_plus(nn.Module):
    def __init__(self, in_channels, num_classes, backend='resnet18', os=16, pretrained='imagenet'):
        '''
        :param in_channels:
        :param num_classes:
        :param backend: only support resnet, otherwise need to have low_features
                        and high_features methods for out
        '''
        super(DeepLabv3_plus, self).__init__()
        self.in_channes = in_channels
        self.num_classes = num_classes
        if hasattr(backend, 'low_features') and hasattr(backend, 'high_features') \
                and hasattr(backend, 'lastconv_channel'):
            self.backend = backend
        elif 'resnet' in backend:
            self.backend = ResnetBackend(backend, os, pretrained)
        elif 'resnext' in backend:
            self.backend = ResnetBackend(backend, os=None, pretrained=pretrained)
        elif 'mobilenet' in backend:
            self.backend = MobileNetBackend(backend, os=os, pretrained=pretrained)
        elif 'shufflenet' in backend:
            self.backend = ShuffleNetBackend(backend, os=os, pretrained=pretrained)
        else:
            raise NotImplementedError

        if hasattr(self.backend, 'interconv_channel'):
            self.aspp_out_channel = self.backend.interconv_channel
        else:
            self.aspp_out_channel = self.backend.lastconv_channel // 8

        self.aspp_pooling = ASPPBlock(self.backend.lastconv_channel, self.aspp_out_channel, os)

        self.cbr_low = nn.Sequential(nn.Conv2d(self.aspp_out_channel, self.aspp_out_channel // 5,
                                               kernel_size=1, bias=False),
                                     SyncBN2d(self.aspp_out_channel // 5),
                                     nn.ReLU(inplace=True))
        self.cbr_last = nn.Sequential(nn.Conv2d(self.aspp_out_channel + self.aspp_out_channel // 5,
                                                self.aspp_out_channel, kernel_size=3, padding=1, bias=False),
                                      SyncBN2d(self.aspp_out_channel),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.aspp_out_channel, self.aspp_out_channel,
                                                kernel_size=3, padding=1, bias=False),
                                      SyncBN2d(self.aspp_out_channel),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.aspp_out_channel, self.num_classes, kernel_size=1))

    def forward(self, x):
        h, w = x.size()[2:]
        low_features, x = self.backend(x)
        x = self.aspp_pooling(x)
        x = F.interpolate(x, size=low_features.size()[2:], mode='bilinear', align_corners=True)
        low_features = self.cbr_low(low_features)

        x = torch.cat([x, low_features], dim=1)
        x = self.cbr_last(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        # print('x.shape',x.shape)
        return x
    #
    # def get_1x_lr_params(self):
    #     modules = [self.backend]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if isinstance(m[1], nn.Conv2d):
    #                 for p in m[1].parameters():
    #                     if p.requires_grad:
    #                         yield p
    #
    # def get_10x_lr_params(self):
    #     modules = [self.aspp_pooling, self.cbr_low, self.cbr_last]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if isinstance(m[1], nn.Conv2d):
    #                 for p in m[1].parameters():
    #                     if p.requires_grad:
    #                         yield p


class ResnetBackend(nn.Module):
    def __init__(self, backend='resnet18', os=16, pretrained='imagenet'):
        '''
        :param backend: resnet<> or se_resnet
        '''
        super(ResnetBackend, self).__init__()
        _all_resnet_models = backbone._all_resnet_backbones
        if backend not in _all_resnet_models:
            raise Exception(f"{backend} must in {_all_resnet_models}")

        if 'se' in backend:
            _backend_model = backbone.__dict__[backend](pretrained=pretrained)
        else:
            _backend_model = backbone.__dict__[backend](pretrained=pretrained, output_stride=os)

        if 'se' in backend:
            self.low_features = nn.Sequential(_backend_model.layer0,
                                              _backend_model.layer1)
        else:
            self.low_features = nn.Sequential(_backend_model.conv1,
                                              _backend_model.bn1,
                                              _backend_model.relu,
                                              _backend_model.maxpool,
                                              _backend_model.layer1
                                              )

        self.high_features = nn.Sequential(_backend_model.layer2,
                                           _backend_model.layer3,
                                           _backend_model.layer4)

        if backend in ['resnet18', 'resnet34']:
            self.lastconv_channel = 512
        else:
            self.lastconv_channel = 512 * 4

    def forward(self, x):
        low_features = self.low_features(x)
        x = self.high_features(low_features)
        return low_features, x


class MobileNetBackend(nn.Module):
    def __init__(self, backend='mobilenet_v2', os=16, pretrained='imagenet', width_mult=1.):
        '''
        :param backend: mobilenet_<>
        '''
        super(MobileNetBackend, self).__init__()
        _all_mobilenet_backbones = backbone._all_mobilenet_backbones
        if backend not in _all_mobilenet_backbones:
            raise Exception(f"{backend} must in {_all_mobilenet_backbones}")

        _backend_model = backbone.__dict__[backend](pretrained=pretrained, output_stride=os, width_mult=width_mult)

        self.low_features = _backend_model.features[:4]

        self.high_features = _backend_model.features[4:]

        self.lastconv_channel = _backend_model.lastconv_channel
        self.interconv_channel = _backend_model.interconv_channel

    def forward(self, x):
        low_features = self.low_features(x)
        x = self.high_features(low_features)
        return low_features, x


class ShuffleNetBackend(nn.Module):
    def __init__(self, backend='shufflenet_v2', os=16, pretrained='imagenet', width_mult=1.):
        '''
        :param backend: mobilenet_<>
        '''
        super(ShuffleNetBackend, self).__init__()
        _all_shufflenet_backbones = backbone._all_shufflenet_backbones
        if backend not in _all_shufflenet_backbones:
            raise Exception(f"{backend} must in {_all_shufflenet_backbones}")

        _backend_model = backbone.__dict__[backend](pretrained=pretrained, output_stride=os, width_mult=width_mult)

        self.low_features = nn.Sequential(_backend_model.conv1,
                                          _backend_model.maxpool)

        self.high_features = _backend_model.features

        self.lastconv_channel = _backend_model.lastconv_channel
        self.interconv_channel = _backend_model.interconv_channel

    def forward(self, x):
        low_features = self.low_features(x)
        x = self.high_features(low_features)
        return low_features, x


if __name__ == '__main__':
    from torchsummary import summary

    deeplabv3_ = DeepLabv3_plus(in_channels=3, num_classes=21, backend='mobilenet_v2', os=16).cuda()
    print(summary(deeplabv3_, [3, 512, 512]))
    print('Total params: ', sum(p.numel() for p in deeplabv3_.parameters() if p.requires_grad))
    x = torch.randn(2, 3, 512, 512)
    out = deeplabv3_(x.cuda())
    print('out.size',out.size())
