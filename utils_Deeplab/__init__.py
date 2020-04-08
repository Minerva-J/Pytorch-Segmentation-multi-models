# -*- coding: utf-8 -*-
from .metrics import AveMeter, ScoreMeter,Timer
from .sync_batchnorm import patch_replication_callback
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d as SyncBN2d
