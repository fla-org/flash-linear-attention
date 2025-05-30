# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Code adapted from https://github.com/mayank31398/cute-kernels

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_lens
from fla.utils import input_guard


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4, 8, 16, 32]
    ],
    key=['D', 'PADDING_SIDE', 'PACK']
)
@triton.jit
def packunpack_sequence_kernel(
    x,
    y,
    cu_seqlens,
    S,
    D,
    BD: tl.constexpr,
    PADDING_SIDE: tl.constexpr,
    PACK: tl.constexpr,
):
    i_d, i_s, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos, eos = tl.load(cu_seqlens + i_b), tl.load(cu_seqlens + i_b + 1)

    T = eos - bos
    if PADDING_SIDE == 'left':
        NP = S - T
        if i_s < NP:
            return
        i_t = bos + (i_s - NP)
    else:
        if i_s >= T:
            return
        i_t = bos + i_s

    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    if PACK:
        b_x = tl.load(x + (i_b * S + i_s) * D + o_d, mask=mask)
        tl.store(y + i_t * D + o_d, b_x, mask=mask)
    else:
        b_x = tl.load(x + i_t * D + o_d, mask=mask)
        tl.store(y + (i_b * S + i_s) * D + o_d, b_x, mask=mask)


def pack_sequence_fwdbwd(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
) -> torch.Tensor:
    B, S = x.shape[:2]
    D = x.numel() // (B * S)
    BD = min(triton.next_power_of_2(D), 4096)
    ND = triton.cdiv(D, BD)

    y = torch.empty(cu_seqlens[-1].item(), *x.shape[2:], device=x.device, dtype=x.dtype)
    packunpack_sequence_kernel[ND, S, B](
        x=x,
        y=y,
        cu_seqlens=cu_seqlens,
        S=S,
        D=D,
        BD=BD,
        PADDING_SIDE=padding_side,
        PACK=True,
    )
    return y


def unpack_sequence_fwdbwd(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
    desired_shape: torch.Size,
) -> torch.Tensor:
    if desired_shape is None:
        desired_shape = (len(cu_seqlens) - 1, prepare_lens(cu_seqlens).max().item(), *x.shape[1:])
    y = torch.zeros(desired_shape, device=x.device, dtype=x.dtype)
    B, S = y.shape[:2]
    D = y.numel() // (B * S)
    BD = min(triton.next_power_of_2(D), 4096)
    ND = triton.cdiv(D, BD)

    packunpack_sequence_kernel[ND, S, B](
        x=x,
        y=y,
        cu_seqlens=cu_seqlens,
        S=S,
        D=D,
        BD=BD,
        PADDING_SIDE=padding_side,
        PACK=False,
    )
    return y


class PackSequenceFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        padding_side: str,
    ) -> torch.Tensor:
        assert padding_side in ['left', 'right']
        assert x.ndim >= 2

        ctx.cu_seqlens = cu_seqlens
        ctx.padding_side = padding_side
        ctx.desired_shape = x.shape

        y = pack_sequence_fwdbwd(
            x=x,
            cu_seqlens=cu_seqlens,
            padding_side=padding_side,
        )
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor | None]:
        dx = unpack_sequence_fwdbwd(
            x=dy,
            cu_seqlens=ctx.cu_seqlens,
            padding_side=ctx.padding_side,
            desired_shape=ctx.desired_shape,
        )
        return dx, *[None] * 10


class UnpackSequenceFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        padding_side: str,
        desired_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        assert padding_side in ['left', 'right']
        assert x.ndim >= 2
        if desired_shape is not None:
            assert desired_shape[0] == cu_seqlens.shape[0] - 1
            assert desired_shape[2:] == x.shape[1:]

        ctx.cu_seqlens = cu_seqlens
        ctx.padding_side = padding_side

        y = unpack_sequence_fwdbwd(
            x=x,
            cu_seqlens=cu_seqlens,
            padding_side=padding_side,
            desired_shape=desired_shape,
        )
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor | None]:
        dx = pack_sequence_fwdbwd(
            x=dy,
            cu_seqlens=ctx.cu_seqlens,
            padding_side=ctx.padding_side,
        )
        return dx, None, None, None


def pack_sequence(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str = 'left'
) -> torch.Tensor:
    return PackSequenceFunction.apply(
        x,
        cu_seqlens,
        padding_side,
    )


def unpack_sequence(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str = 'left',
    desired_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
    return UnpackSequenceFunction.apply(
        x,
        cu_seqlens,
        padding_side,
        desired_shape,
    )
