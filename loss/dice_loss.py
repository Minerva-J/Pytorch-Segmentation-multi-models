# -*- coding: utf-8 -*-
import torch
from torch import nn


class DiceLoss:
    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, input, target):
        '''
        :param input: [batch_size,h,w]
        :param target: [batch_size,h,w]
        :return: loss
        '''
        input = torch.sigmoid(input)
        smooth = 1.

        iflat = input.view(-1).float()
        tflat = target.view(-1).float()
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))

if __name__ == '__main__':
    diceloss=DiceLoss()
    print(diceloss(torch.randn(3,4,4),torch.empty(3,4,4).random_(1)))