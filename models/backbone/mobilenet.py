# -*- coding: utf-8 -*-
# Thanks to https://github.com/tonylins/pytorch-mobilenet-v2
import torch
import torch.nn as nn
import math
from utils_Deeplab import SyncBN2d

__all__ = ['MobileNetV2', 'mobilenet_v2']


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        SyncBN2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        SyncBN2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, dilation, dilation, groups=hidden_dim, bias=False),
                SyncBN2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                SyncBN2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                SyncBN2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, dilation, dilation, groups=hidden_dim, bias=False),
                SyncBN2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                SyncBN2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1., pretrained=True, output_stride=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        # last_channel = 1280

        if output_stride == 16:
            interverted_residual_setting = \
                [
                    # t, c, n, s, d
                    [1, 16, 1, 1, 1],
                    [6, 24, 2, 2, 1],
                    [6, 32, 3, 2, 1],
                    [6, 64, 4, 2, 1],
                    [6, 96, 3, 1, 2],
                    [6, 160, 3, 1, 4],
                    [6, 320, 1, 1, 8]
                ]
        elif output_stride == 8:
            interverted_residual_setting = \
                [
                    # t, c, n, s, d
                    [1, 16, 1, 1, 1],
                    [6, 24, 2, 2, 1],
                    [6, 32, 3, 2, 1],
                    [6, 64, 4, 1, 2],
                    [6, 96, 3, 1, 4],
                    [6, 160, 3, 1, 8],
                    [6, 320, 1, 1, 16]
                ]
        elif output_stride is None:
            interverted_residual_setting = \
                [
                    # t, c, n, s, d
                    [1, 16, 1, 1, 1],
                    [6, 24, 2, 2, 1],
                    [6, 32, 3, 2, 1],
                    [6, 64, 4, 2, 1],
                    [6, 96, 3, 1, 1],
                    [6, 160, 3, 2, 1],
                    [6, 320, 1, 1, 1]
                ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        # self.lastconv_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s, d in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, d, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, d, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.interconv_channel = 24
        self.lastconv_channel = input_channel
        # self.features.append(conv_1x1_bn(input_channel, self.lastconv_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, n_class),
        # )

        self._initialize_weights()

        # if pretrained:
            # self._load_pretrained_model(torch.load('/home/yhuangcc/ImageSegmentation/checkpoints/mobilenet_v2.pth.tar'))

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, SyncBN2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _load_pretrained_model(self, pretrain_dict):
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def mobilenet_v2(pretrained=False, output_stride=None, **kwargs):
    model = MobileNetV2(pretrained=pretrained, output_stride=output_stride, **kwargs)
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = MobileNetV2(pretrained=True, output_stride=None)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, (3, 224, 224))
    print('Total params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
