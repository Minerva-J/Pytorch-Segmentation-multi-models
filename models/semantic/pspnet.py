# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from models import backbone
from modules import PPBlock
from utils_Deeplab import SyncBN2d


class PSPNet(nn.Module):
    def __init__(self, in_channels, num_classes, backend='resnet18', pool_scales=(1, 2, 3, 6), pretrained='imagenet'):
        '''
        :param in_channels:
        :param num_classes:
        :param backend: only support resnet, otherwise need to have low_features for aux
                        and high_features methods for out
        '''
        super(PSPNet, self).__init__()
        self.in_channes = in_channels
        self.num_classes = num_classes

        if hasattr(backend, 'low_features') and hasattr(backend, 'high_features') \
                and hasattr(backend, 'lastconv_channel'):
            self.backend = backend
        elif 'resnet' in backend:
            ## in the future, I may retrain on os=16 (now without os)
            self.backend = ResnetBackend(backend, os=None, pretrained=pretrained)
        elif 'resnext' in backend:
            self.backend = ResnetBackend(backend, os=None, pretrained=pretrained)
        else:
            raise NotImplementedError

        self.pp_out_channel = 256
        self.pyramid_pooling = PPBlock(self.backend.lastconv_channel, out_channel=self.pp_out_channel,
                                       pool_scales=pool_scales)

        self.cbr_last = nn.Sequential(nn.Conv2d(self.backend.lastconv_channel + self.pp_out_channel * len(pool_scales),
                                                self.pp_out_channel, kernel_size=3, padding=1, bias=False),
                                      SyncBN2d(self.pp_out_channel),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(self.pp_out_channel, num_classes, kernel_size=1))
        self.cbr_deepsup = nn.Sequential(nn.Conv2d(self.backend.lastconv_channel // 2,
                                                   self.backend.lastconv_channel // 4,
                                                   kernel_size=3, padding=1, bias=False),
                                         SyncBN2d(self.backend.lastconv_channel // 4),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(0.1),
                                         nn.Conv2d(self.backend.lastconv_channel // 4,
                                                   num_classes, kernel_size=1))
        # self.LS = nn.LogSoftmax(dim=1)
    def forward(self, x):
        h, w = x.size()[2:]
        aux, x = self.backend(x)

        x = self.pyramid_pooling(x)
        x = self.cbr_last(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        if self.training:
            aux = self.cbr_deepsup(aux)
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            return aux, x
            # return self.LS(aux), self.LS(x)
        else:
            # return self.LS(x)
            return x

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
    #     modules = [self.pyramid_pooling, self.cbr_deepsup, self.cbr_last]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if isinstance(m[1], nn.Conv2d):
    #                 for p in m[1].parameters():
    #                     if p.requires_grad:
    #                         yield p


class ResnetBackend(nn.Module):
    def __init__(self, backend='resnet18', os=16, pretrained='imagenet'):
        '''
        :param backend: resnet<> or se_resnet<>
        '''
        super(ResnetBackend, self).__init__()
        _all_resnet_models = backbone._all_resnet_backbones
        if backend not in _all_resnet_models:
            raise NotImplementedError(f"{backend} must in {_all_resnet_models}")

        if 'se' in backend:
            _backend_model = backbone.__dict__[backend](pretrained=pretrained)
        else:
            _backend_model = backbone.__dict__[backend](pretrained=pretrained, output_stride=os)

        if 'se' in backend:
            self.low_features = nn.Sequential(_backend_model.layer0,
                                              _backend_model.layer1,
                                              _backend_model.layer2,
                                              _backend_model.layer3)
        else:
            self.low_features = nn.Sequential(_backend_model.conv1,
                                              _backend_model.bn1,
                                              _backend_model.relu,
                                              _backend_model.maxpool,
                                              _backend_model.layer1,
                                              _backend_model.layer2,
                                              _backend_model.layer3)

        self.high_features = nn.Sequential(_backend_model.layer4)

        if backend in ['resnet18', 'resnet34']:
            self.lastconv_channel = 512
        else:
            self.lastconv_channel = 512 * 4

    def forward(self, x):
        low_features = self.low_features(x)
        x = self.high_features(low_features)
        return low_features, x


if __name__ == '__main__':
    from torchsummary import summary

    pspnet = PSPNet(in_channels=3, num_classes=21, backend='se_resnext50_32x4d', pool_scales=(1, 2, 3, 6)).cuda()
    print(summary(pspnet, [3, 224, 224]))
    print('Total params: ', sum(p.numel() for p in pspnet.parameters() if p.requires_grad))
    x = torch.randn(2, 3, 224, 224)
    out_aux, out = pspnet(x.cuda())
    print(out_aux.size())
    print(out.size())
