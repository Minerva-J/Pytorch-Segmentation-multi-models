# -*- coding: utf-8 -*-
import torch
import time
import numpy as np
from models import DeepLabv3_plus
from models.backbone import shufflenet_v2, mobilenet_v2


def get_time(model, h, w):
    run_time = list()
    for i in range(0, 100):
        input = torch.randn(1, 3, h, w).cuda()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input)
            torch.cuda.synchronize()  # wait for mm to finish
            end = time.perf_counter()
            run_time.append(end - start)
    run_time.pop(0)
    print('Mean running time is ', np.mean(run_time))


m1 = DeepLabv3_plus(3, 19, 'mobilenet_v2').cuda().eval()
m2 = DeepLabv3_plus(3, 19, 'shufflenet_v2').cuda().eval()
m3 = mobilenet_v2().cuda().eval()
m4 = shufflenet_v2().cuda().eval()
get_time(m1, 512, 512)
get_time(m2, 512, 512)
get_time(m3, 512, 512)
get_time(m4, 512, 512)
get_time(m1, 512, 1024)
get_time(m2, 512, 1024)
get_time(m3, 512, 1024)
get_time(m4, 512, 1024)
