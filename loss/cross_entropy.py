# -*- coding: utf-8 -*-
import torch
from torch import nn


class CrossEntropyLoss:
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self._backend_loss = nn.CrossEntropyLoss(self.weight,
                                                 ignore_index=self.ignore_index,
                                                 reduction=self.reduction)

    def __call__(self, input, target, scale=[0.4, 1.]):
        '''
        :param input: [batch_size,c,h,w]
        :param target: [batch_size,h,w]
        :param scale: [...]
        :return: loss
        '''
        if isinstance(input, tuple) and (scale is not None):
            loss = 0
            for i, inp in enumerate(input):
                loss += scale[i] * self._backend_loss(inp, target)
            return loss
        else:
            return self._backend_loss(input, target)


if __name__ == '__main__':
    criterion = CrossEntropyLoss()
    print(criterion(torch.randn(4, 5, 6, 6), torch.empty(4, 6, 6).random_(5).long()))
    print(criterion((torch.randn(4, 5, 6, 6), torch.randn(4, 5, 6, 6)),
                    torch.empty(4, 6, 6).random_(5).long(), [1, 0.4]))
