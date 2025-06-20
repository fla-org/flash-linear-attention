# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import is_amd

NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if is_amd else [1, 2, 4, 8, 16, 32]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit
def softmax_fwd_kernel(
    x,
    p,
    D: tl.constexpr,
    B: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    b_m = tl.max(b_x, 0)
    b_x = exp(b_x - b_m)
    b_p = b_x / tl.sum(b_x, 0)

    tl.store(p + i_n * D + o_d, b_p.to(p.dtype.element_ty), mask=m_d)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit
def softmax_bwd_kernel(
    p,
    dp,
    ds,
    D: tl.constexpr,
    B: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < D

    b_p = tl.load(p + i_n * D + o_d, mask=m_d, other=0.)
    b_dp = tl.load(dp + i_n * D + o_d, mask=m_d, other=0.)
    b_pp = tl.sum(b_p * b_dp, 0)
    b_ds = b_p * b_dp - b_p * b_pp
    tl.store(ds + i_n * D + o_d, b_ds.to(ds.dtype.element_ty), mask=m_d)


def softmax_fwd(
    x: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    shape = x.shape
    x = x.view(-1, x.shape[-1])

    N, D = x.shape
    B = triton.next_power_of_2(D)

    p = torch.empty_like(x, dtype=dtype)
    softmax_fwd_kernel[(N,)](
        x=x,
        p=p,
        D=D,
        B=B
    )
    return p.view(*shape)


def softmax_bwd(
    p: torch.Tensor,
    dp: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    shape = p.shape
    p = p.view(-1, p.shape[-1])
    ds = torch.empty_like(p, dtype=dtype)

    N, D = p.shape
    B = triton.next_power_of_2(D)
    softmax_bwd_kernel[(N,)](
        p=p,
        dp=dp,
        ds=ds,
        D=D,
        B=B
    )
    return ds.view(*shape)
