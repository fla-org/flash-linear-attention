# -*- coding: utf-8 -*-

import torch


@torch.jit.script
def normalize_output(q, k, o):
    k = k.cumsum(1)
    z = (q * k).sum(-1, keepdim=True)
    return o / (z + 1e-10)
