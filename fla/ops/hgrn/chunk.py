# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# this function implements the chunkwise form of HGRN, inspired by
# [Volodymyr Kyrylov in his blog post](https://proger.github.io/posts/scan/chunk.html)
# also refer to the `accelerated-scan` lib: https://github.com/proger/accelerated-scan

# from tests on H800, with B, D = 16, 128, we see that the chunk can be greatly faster than the recurrent:
#
# Performance:
#    seq_len     chunk  recurrent  chunk_bwd  recurrent_bwd
# 0    128.0  0.039360   0.061056   0.312160       0.205008
# 1    256.0  0.045824   0.123712   0.308784       0.297696
# 2    512.0  0.058688   0.241952   0.310720       0.626528
# 3   1024.0  0.088288   0.476992   0.313184       1.333152
# 4   2048.0  0.169472   0.943264   0.452464       2.724864
# 5   4096.0  0.329920   1.886144   0.881600       5.551520
# 6   8192.0  0.647872   3.755040   1.740496      11.117184
# 7  16384.0  1.272064   7.520576   3.446608      22.362528

from typing import Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import input_guard


@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def chunk_hgrn_fwd_kernel_h(
    x,
    g,
    gc,
    o,
    h0,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x = x + i_b * T * D + i_t * BT * D + o_d
    p_g = g + i_b * T * D + i_t * BT * D + o_d
    p_gc = gc + i_b * T * D + i_t * BT * D + o_d
    p_o = o + i_b * T * D + i_t * BT * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    b_gc = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if i_t == 0:
            b_h += tl.load(h0 + i_b * D + o_d, mask=mask, other=0).to(tl.float32)
    for i in range(0, BT):
        mask_t = mask & ((i_t * BT + i) < T)
        b_x = tl.load(p_x, mask=mask_t, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask_t, other=0).to(tl.float32)
        b_h = exp(b_g) * b_h + b_x
        b_gc = b_gc + b_g
        tl.store(p_gc, b_gc.to(p_o.dtype.element_ty), mask=mask_t)
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask_t)

        p_x += D
        p_g += D
        p_gc += D
        p_o += D


@triton.jit(do_not_specialize=['T'])
def chunk_hgrn_fwd_kernel_o(
    gc,
    o,
    s_b,
    s_t,
    s_d,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    for i_t in range(1, tl.cdiv(T, BT)):
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))

        # [BD,]
        b_h0 = tl.load(o + i_b * T * D + i_t * BT * D - D + o_d, mask=mask, other=0).to(tl.float32)
        # [BT, BD]
        b_gc = tl.load(p_gc, boundary_check=(0, 1)).to(tl.float32)
        b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
        b_o = b_o + exp(b_gc) * b_h0[None, :]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def chunk_hgrn_bwd_kernel_h(
    g,
    gc,
    dx,
    do,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    BC = min(BT, T - i_t * BT)
    NT = tl.num_programs(1)

    p_g = g + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_gc = gc + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_dx = dx + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_do = do + (i_b * T + i_t * BT + BC - 1) * D + o_d

    if i_t == NT - 1:
        b_gc = tl.zeros([BD], dtype=tl.float32)
    else:
        b_gc = tl.load(g + (i_b * T + i_t * BT + BT) * D + o_d, mask=mask, other=0).to(tl.float32)
    b_dh = tl.zeros([BD], dtype=tl.float32)
    for _ in range(BC - 1, -1, -1):
        tl.store(p_gc, b_gc.to(p_gc.dtype.element_ty), mask=mask)

        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)

        b_gc = b_gc + b_g
        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * exp(b_g)

        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)

        p_g -= D
        p_gc -= D
        p_dx -= D
        p_do -= D


@triton.jit(do_not_specialize=['T'])
def chunk_hgrn_bwd_kernel_o(
    g,
    gc,
    o,
    dx,
    dg,
    s_b,
    s_t,
    s_d,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_g = tl.make_block_ptr(g + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT - 1, i_d * BD), (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))

        # [BD,]
        mask_t = mask & ((i_t + 1) * BT < T)
        b_ht = tl.load(dx + i_b * T * D + (i_t + 1) * BT * D + o_d, mask=mask_t, other=0).to(tl.float32)
        # [BT, BD]
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_gc = tl.load(p_gc, boundary_check=(0, 1)).to(tl.float32)
        b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
        b_dx = tl.load(p_dx, boundary_check=(0, 1)).to(tl.float32)

        b_dx = b_dx + exp(b_gc) * b_ht[None, :]
        b_dg = b_o * b_dx * exp(b_g)
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


class ChunkHGRNFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        B, T, D = x.shape
        BT, BD = 128, min(64, triton.next_power_of_2(D))
        num_warps = 8 if BD == 64 else 4

        gc = torch.empty_like(g, dtype=torch.float)
        o = torch.empty_like(x, dtype=torch.float)
        def grid(meta): return (triton.cdiv(D, meta['BD']), triton.cdiv(T, meta['BT']), B)
        chunk_hgrn_fwd_kernel_h[grid](
            x, g, gc, o, initial_state,
            T=T, D=D, BT=BT,
            USE_INITIAL_STATE=initial_state is not None
        )
        def grid(meta): return (triton.cdiv(D, meta['BD']), B)
        chunk_hgrn_fwd_kernel_o[grid](
            gc, o,
            o.stride(-3), o.stride(-2), o.stride(-1),
            T=T, D=D, BT=BT, BD=BD,
            num_warps=num_warps
        )
        final_state = None
        if output_final_state:
            final_state = o[:, -1].clone()
        o = o.to(x.dtype)
        ctx.save_for_backward(g, o, initial_state)
        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht=None):
        g, o, initial_state = ctx.saved_tensors
        B, T, D = do.shape
        BT, BD = 128, min(64, triton.next_power_of_2(D))
        num_warps = 8 if BD == 64 else 4

        gc = torch.empty_like(g, dtype=torch.float)
        dx = torch.empty_like(o, dtype=torch.float)
        def grid(meta): return (triton.cdiv(D, meta['BD']), triton.cdiv(T, meta['BT']), B)
        chunk_hgrn_bwd_kernel_h[grid](
            g, gc, dx, do,
            T=T, D=D, BT=BT
        )

        dg = torch.empty_like(g, dtype=torch.float)
        def grid(meta): return (triton.cdiv(D, meta['BD']), B)
        chunk_hgrn_bwd_kernel_o[grid](
            g, gc, o, dx, dg,
            o.stride(-3), o.stride(-2), o.stride(-1),
            T=T, D=D, BT=BT, BD=BD,
            num_warps=num_warps
        )
        if initial_state is not None:
            dg[:, 0] = (initial_state * dx[:, 0] * g[:, 0].float().exp()).to(dg.dtype)

        return dx.to(o.dtype), dg, None, None


@torch.compiler.disable
def chunk_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ChunkHGRNFunction.apply(x, g, initial_state, output_final_state)
