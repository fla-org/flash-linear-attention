# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang

# code adapted from
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.utils import input_guard


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BM`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.heuristics({
    'HAS_INPUT': lambda args: args['input'] is not None,
    'HAS_ALPHA': lambda args: args['alpha'] is not None,
    'HAS_BETA': lambda args: args['beta'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BK': 64, 'BN': 256, 'G': 4}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64, 'BK': 32, 'BN': 256, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BK': 32, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BK': 32, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BK': 32, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BK': 32, 'BN': 32, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BK': 32, 'BN': 32, 'G': 4}, num_stages=5, num_warps=2),
        triton.Config({'BM': 32, 'BK': 32, 'BN': 64, 'G': 4}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BM': 128, 'BK': 128, 'BN': 256, 'G': 4}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BK': 128, 'BN': 128, 'G': 4}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BK': 128, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BK': 128, 'BN': 256, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BK': 128, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BK': 64, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BK': 64, 'BN': 32, 'G': 4}, num_stages=4, num_warps=4)
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a,
    b,
    c,
    input,
    alpha,
    beta,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `s_am` is how much to increase `a`
    # by to get the element one row down (A has M rows).
    s_am,
    s_ak,
    s_bk,
    s_bn,
    s_cm,
    s_cn,
    # Meta-parameters
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    G: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_INPUT: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    HAS_BETA: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    NM, NN = tl.num_programs(0), tl.num_programs(1)
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    i_m, i_n = tl.swizzle2d(i_m, i_n, NM, NN, G)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `p_a` is a block of [BM, BK] pointers
    # `p_b` is a block of [BK, BN] pointers
    # See above `Pointer Arithmetic` section for details
    o_am = (i_m * BM + tl.arange(0, BM)) % M
    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)

    p_a = a + (o_am[:, None] * s_am + o_k[None, :] * s_ak)
    p_b = b + (o_k[:, None] * s_bk + o_bn[None, :] * s_bn)

    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)
        # We accumulate along the K dimension.
        b_acc += tl.dot(b_a, b_b, allow_tf32=False)
        # Advance the ptrs to the next K block.
        p_a += BK * s_ak
        p_b += BK * s_bk

    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    mask = (o_cm[:, None] < M) & (o_cn[None, :] < N)

    b_c = b_acc
    # You can fuse arbitrary activation functions here
    # while the b_acc is still in FP32!
    if ACTIVATION == "leaky_relu":
        b_c = leaky_relu(b_c)
    if HAS_ALPHA:
        b_c *= tl.load(alpha)
    if HAS_INPUT:
        p_i = input + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]
        b_i = tl.load(p_i, mask=mask, other=0.0).to(tl.float32)
        if HAS_BETA:
            b_i *= tl.load(beta)
        b_c += b_i

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    p_c = c + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]
    tl.store(p_c, b_c.to(c.dtype.element_ty), mask=mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@input_guard
def matmul(a, b, activation=''):
    assert a.shape[1] == b.shape[0], 'Incompatible dimensions (A: {}x{}, B: {}x{})'.format(*a.shape, *b.shape)

    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = a.new_empty(M, N)
    # 1D launch kernel where each block gets its own program.

    def grid(meta): return (triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))
    matmul_kernel[grid](
        a, b, c, None, None, None,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c


@input_guard
def addmm(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    inplace: Optional[bool] = False
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], 'Incompatible dimensions (A: {}x{}, B: {}x{})'.format(*a.shape, *b.shape)

    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = x if inplace else a.new_empty(M, N)

    def grid(meta): return (triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))
    matmul_kernel[grid](
        a, b, c, x, alpha, beta,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=None,
    )
    return c
