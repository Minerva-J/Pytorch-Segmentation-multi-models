# -*- coding: utf-8 -*-
import torch
from torch import nn


class FocalLoss:
    def __init__(self, weight=None, gamma=2, alpha=0.5, ignore_index=255):
        self.weight = weight
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.alpha = alpha
        self._backend_loss = nn.CrossEntropyLoss(self.weight,
                                                 ignore_index=self.ignore_index,
                                                 reduction='none')

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
                loss += scale[i] * self.focalloss(inp, target)
            return loss
        else:
            return self.focalloss(input, target)

    def focalloss(self, input, target):
        logpt = -self._backend_loss(input, target)
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


if __name__ == '__main__':
    criterion = FocalLoss()
    print(criterion(torch.randn(4, 5, 6, 6), torch.empty(4, 6, 6).random_(5).long()))
    print(criterion((torch.randn(4, 5, 6, 6), torch.randn(4, 5, 6, 6)),
                    torch.empty(4, 6, 6).random_(5).long(), [1, 0.4]))
