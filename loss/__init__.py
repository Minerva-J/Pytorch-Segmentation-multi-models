# -*- coding: utf-8 -*-
from .cross_entropy import CrossEntropyLoss
from .dice_loss import DiceLoss
from .focal import FocalLoss
from .lovasz import LovaszLoss

LossSelector = {'ce': CrossEntropyLoss,
                'dice': DiceLoss,
                'focal': FocalLoss,
                'lovasz': LovaszLoss}
