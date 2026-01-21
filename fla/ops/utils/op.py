# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from fla.utils import IS_GATHER_SUPPORTED

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    @triton.jit
    def exp(x): return tldevice.fast_expf(x.to(tl.float32))
    @triton.jit
    def exp2(x): return tldevice.exp2(x.to(tl.float32))
    @triton.jit
    def log(x): return tldevice.fast_logf(x.to(tl.float32))
    @triton.jit
    def log2(x): return tldevice.fast_log2f(x.to(tl.float32))
else:
    @triton.jit
    def exp(x): return tl.exp(x.to(tl.float32))
    @triton.jit
    def exp2(x): return tl.math.exp2(x.to(tl.float32))
    @triton.jit
    def log(x): return tl.log(x.to(tl.float32))
    @triton.jit
    def log2(x): return tl.log2(x.to(tl.float32))


if not IS_GATHER_SUPPORTED:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None
else:
    gather = tl.gather


if hasattr(triton.language, '_experimental_make_tensor_descriptor'):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, 'make_tensor_descriptor'):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """
    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None
