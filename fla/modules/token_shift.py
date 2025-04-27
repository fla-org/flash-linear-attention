# -*- coding: utf-8 -*-

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.utils import input_guard


def token_shift_ref(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if cu_seqlens is not None:
        # Variable length mode with cu_seqlens
        assert x.dim() == 3, "Input must be [B, T, D]"
        B, T, D = x.shape
        assert B == 1, "Batch size must be 1 when using cu_seqlens"

        result = torch.zeros_like(x)
        N = cu_seqlens.shape[0] - 1

        for i in range(N):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i+1].item()
            seq_len = end - start

            if seq_len <= 1:
                # For sequences of length 1 or 0, delta is simply -x
                result[0, start:end] = -x[0, start:end]
            else:
                # For longer sequences, handle padding manually
                shifted = torch.zeros_like(x[0, start:end])
                shifted[1:] = x[0, start:end-1]
                delta = shifted - x[0, start:end]
                result[0, start:end] = delta

        return result
    else:
        time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))
        shifted = time_shift(x)
        delta = shifted - x
        return delta


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['BD'],
)
@triton.jit
def token_shift_fwd_kernel(
    x,
    y,
    cu_seqlens,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)

        if i_t < bos or i_t >= eos:
            return

        is_first_pos = (i_t - bos == 0)
    else:
        is_first_pos = (i_t == 0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        base_offset = i_t * D + o_d
    else:
        base_offset = i_b * T*D + i_t * D + o_d

    b_x = tl.load(x + base_offset, mask=m_d)

    if is_first_pos:
        # First position in sequence: delta = -hidden_states
        tl.store(y + base_offset, -b_x, mask=m_d)
    else:
        # Other positions: delta = prev - curr
        if IS_VARLEN:
            prev_offset = (i_t - 1) * D + o_d
        else:
            prev_offset = i_b * T*D + (i_t-1) * D + o_d

        prev_values = tl.load(x + prev_offset, mask=m_d)
        delta = prev_values - b_x
        tl.store(y + base_offset, delta, mask=m_d)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['D'],
)
@triton.jit
def token_shift_bwd_kernel(
    dx,
    dy,
    cu_seqlens,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)

        if i_t < bos or i_t >= eos:
            return

        local_pos = i_t - bos
        is_last_pos = (local_pos == eos - bos - 1)
    else:
        is_last_pos = (i_t == T - 1)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        base_offset = i_t * D + o_d
    else:
        base_offset = i_b * T*D + i_t * D + o_d

    b_dy = tl.load(dy + base_offset, mask=m_d)

    if is_last_pos:
        # Last position: b_dx = -grad_delta[t]
        b_dx = -b_dy
    else:
        # Other positions: b_dx = -grad_delta[t] + grad_delta[t+1]
        if IS_VARLEN:
            next_offset = (i_t+1) * D + o_d
        else:
            next_offset = i_b * T*D + (i_t+1) * D + o_d

        b_dx = -b_dy + tl.load(dy + next_offset, mask=m_d)

    tl.store(dx + base_offset, b_dx, mask=m_d)


def token_shift_fwd(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    B, T, D = x.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BD = triton.next_power_of_2(D)

    y = torch.empty_like(x)

    grid = (N, T)
    token_shift_fwd_kernel[grid](
        x=x,
        y=y,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        BD=BD,
    )

    return y


def token_shift_bwd(
    dy: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    B, T, D = dy.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BD = triton.next_power_of_2(D)

    dx = torch.empty_like(dy)

    grid = (N, T)
    token_shift_bwd_kernel[grid](
        dy=dy,
        dx=dx,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        BD=BD,
    )
    return dx


class TokenShift(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None):
        ctx.cu_seqlens = cu_seqlens
        return token_shift_fwd(x, cu_seqlens)

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor):
        cu_seqlens = ctx.cu_seqlens
        dx = token_shift_bwd(dy, cu_seqlens)
        return dx, None


def token_shift(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None
):
    """
    Implementation of token shift using Triton kernels
    Args:
        x: Input tensor of shape [B, T, D]
        cu_seqlens: Cumulative sequence lengths (optional)
    Returns:
        Tensor of same shape as input with token shift applied
    """
    if cu_seqlens is not None:
        assert x.dim() == 3, "Input must be [B, T, D]"
        assert x.shape[0] == 1, "Batch size must be 1 when using cu_seqlens"

    return TokenShift.apply(x, cu_seqlens)
