# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch


@torch.jit.script
def normalize_output(q, k, o, z_state):
    k = k.cumsum(-2)
    k = k + z_state
    z = (q * k).sum(-1, keepdim=True)
    return o / (z + 1e-10), k[...,-1:,:]

