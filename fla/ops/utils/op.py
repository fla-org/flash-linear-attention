# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from fla.utils import is_gather_supported

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    div = tldevice.fast_dividef
    exp = tldevice.fast_expf
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    @triton.jit
    def div_normal(x, y):
        return x / y
    div = div_normal
    exp = tl.exp
    log = tl.log
    log2 = tl.log2


@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float('-inf')))


if not is_gather_supported:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        # This is a fallback implementation when tl.gather is not supported
        # In order to pass triton compiler, there is no actual gather operation
        return src
else:
    gather = tl.gather
