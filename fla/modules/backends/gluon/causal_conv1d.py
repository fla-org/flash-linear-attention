# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Causal depthwise convolution with register reuse and fused backward recomputation."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from fla.ops.utils import prepare_chunk_indices
from fla.utils import input_guard


@gluon.jit(do_not_specialize=['T'])
def causal_conv1d_fwd_kernel_gluon(
    x, weight, bias, residual, initial_state, y, cu_seqlens, chunk_indices,
    T, stride_x_n: gl.constexpr, stride_x_t: gl.constexpr,
    D: gl.constexpr, W: gl.constexpr, BT: gl.constexpr, BD: gl.constexpr, BT_UNROLL: gl.constexpr,
    NUM_WARPS: gl.constexpr, ACTIVATION: gl.constexpr, NUM_SPLITS: gl.constexpr = 1,
):
    # lanes own adjacent channels; each lane reuses its input window across BT_UNROLL outputs.
    layout: gl.constexpr = gl.BlockedLayout([1, BD // 32], [1, 32], [NUM_WARPS, 1], [1, 0])
    i_d = gl.program_id(0).to(gl.int64)
    i_t = gl.program_id(1).to(gl.int64)
    i_b = gl.program_id(2).to(gl.int64)
    i_split = i_t % NUM_SPLITS
    i_t //= NUM_SPLITS
    if cu_seqlens is not None:
        i_n = gl.load(chunk_indices + 2 * i_t).to(gl.int64)
        i_t = gl.load(chunk_indices + 2 * i_t + 1).to(gl.int64)
        bos = gl.load(cu_seqlens + i_n).to(gl.int64)
        T = gl.load(cu_seqlens + i_n + 1).to(gl.int64) - bos
        p_x = x + bos * stride_x_t
    else:
        i_n = i_b
        bos = i_b * T
        p_x = x + i_b * stride_x_n
    o_d = i_d * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    o_t = (i_t * NUM_SPLITS + i_split) * BT
    o_t += gl.arange(0, BT // BT_UNROLL, layout=gl.SliceLayout(1, layout)).to(gl.int64) * BT_UNROLL
    b_x = ()
    b_w = ()
    for i_w in gl.static_range(W):
        b_w += (gl.load(weight + o_d * W + i_w, o_d < D, other=0).to(gl.float32),)
    for i_x in gl.static_range(BT_UNROLL + W - 1):
        o_x = o_t + i_x - W + 1
        b_xi = gl.load(p_x + o_x[:, None] * stride_x_t + o_d[None, :],
                       (o_x[:, None] >= 0) & (o_x[:, None] < T) & (o_d[None, :] < D), other=0).to(gl.float32)
        if initial_state is not None:
            b_xi += gl.load(initial_state + i_n * D * W + o_d[None, :] * W + (o_x[:, None] + W),
                            (o_x[:, None] < 0) & (o_x[:, None] >= 1 - W) & (o_d[None, :] < D), other=0).to(gl.float32)
        b_x += (b_xi,)
    if bias is not None:
        b_bias = gl.load(bias + o_d, o_d < D, other=0).to(gl.float32)
    for i_r in gl.static_range(BT_UNROLL):
        b_y = gl.full((BT // BT_UNROLL, BD), 0, gl.float32, layout)
        for i_w in gl.static_range(W):
            b_y += b_x[i_r + i_w] * b_w[i_w][None, :]
        if bias is not None:
            b_y += b_bias[None, :]
        if ACTIVATION == 'silu' or ACTIVATION == 'swish':
            b_y *= 1. / (1. + gl.exp(-b_y))
        o_y = (bos + o_t[:, None] + i_r) * D + o_d[None, :]
        m_y = (o_t[:, None] + i_r < T) & (o_d[None, :] < D)
        if residual is not None:
            b_y += gl.load(residual + o_y, m_y, other=0).to(gl.float32)
        gl.store(y + o_y, b_y, m_y)


@input_guard(no_guard_contiguous=['x'])
def causal_conv1d_fwd(
    x, weight, bias=None, residual=None, initial_state=None, output_final_state=False,
    activation=None, cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, BT=64, layout_fallback=False,
):
    B, T, D = x.shape
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    use_small_tile = B * T * D <= 1048576
    BD, num_splits = (32, 2) if use_small_tile else (64, 1)
    num_warps = 4
    causal_conv1d_fwd_kernel_gluon[(triton.cdiv(D, BD), NT * num_splits, B)](
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        y=y,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        stride_x_n=x.stride(0),
        stride_x_t=x.stride(1),
        D=D,
        W=weight.shape[1],
        BT=BT // num_splits,
        BD=BD,
        BT_UNROLL=8,
        NUM_WARPS=num_warps,
        ACTIVATION=activation,
        NUM_SPLITS=num_splits,
        num_warps=num_warps,
    )
    final_state = None
    if output_final_state:
        from fla.modules.conv.triton.ops import causal_conv1d_update_states
        final_state = causal_conv1d_update_states(
            x=x,
            state_len=weight.shape[1],
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )
    return y, final_state


@gluon.jit(do_not_specialize=['T'])
def causal_conv1d_bwd_kernel_gluon(
    x, weight, bias, dy, dx, dw_partial, db_partial, cu_seqlens, chunk_indices,
    T, stride_x_n: gl.constexpr, stride_x_t: gl.constexpr,
    stride_dy_n: gl.constexpr, stride_dy_t: gl.constexpr, stride_dy_d: gl.constexpr,
    D: gl.constexpr, W: gl.constexpr, BT: gl.constexpr, BD: gl.constexpr, BT_UNROLL: gl.constexpr,
    NUM_WARPS: gl.constexpr, ACTIVATION: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, BD // 32], [1, 32], [NUM_WARPS, 1], [1, 0])
    i_d = gl.program_id(0).to(gl.int64)
    i_t = gl.program_id(1).to(gl.int64)
    i_b = gl.program_id(2).to(gl.int64)
    i_tg = i_b * gl.num_programs(1) + i_t
    if cu_seqlens is not None:
        i_n = gl.load(chunk_indices + 2 * i_t).to(gl.int64)
        i_t = gl.load(chunk_indices + 2 * i_t + 1).to(gl.int64)
        bos = gl.load(cu_seqlens + i_n).to(gl.int64)
        T = gl.load(cu_seqlens + i_n + 1).to(gl.int64) - bos
        p_x = x + bos * stride_x_t
        p_dy = dy + bos * stride_dy_t
    else:
        bos = i_b * T
        p_x = x + i_b * stride_x_n
        p_dy = dy + i_b * stride_dy_n
    o_d = i_d * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    o_t = i_t * BT + gl.arange(0, BT // BT_UNROLL, layout=gl.SliceLayout(1, layout)).to(gl.int64) * BT_UNROLL
    b_w = ()
    b_x = ()
    b_dy = ()
    b_dw = ()
    for i_w in gl.static_range(W):
        b_w += (gl.load(weight + o_d * W + i_w, o_d < D, other=0).to(gl.float32),)
        b_dw += (gl.full((BT // BT_UNROLL, BD), 0, gl.float32, layout),)
    if bias is not None:
        b_bias = gl.load(bias + o_d, o_d < D, other=0).to(gl.float32)
    for i_x in gl.static_range(BT_UNROLL + 2 * W - 2):
        o_x = o_t + i_x - W + 1
        b_x += (gl.load(p_x + o_x[:, None] * stride_x_t + o_d[None, :],
                        (o_x[:, None] >= 0) & (o_x[:, None] < T) & (o_d[None, :] < D), other=0).to(gl.float32),)
    for i_r in gl.static_range(BT_UNROLL + W - 1):
        b_dyi = gl.load(p_dy + (o_t[:, None] + i_r) * stride_dy_t + o_d[None, :] * stride_dy_d,
                        (o_t[:, None] + i_r < T) & (o_d[None, :] < D), other=0).to(gl.float32)
        if ACTIVATION == 'silu' or ACTIVATION == 'swish':
            b_y = gl.full((BT // BT_UNROLL, BD), 0, gl.float32, layout)
            for i_w in gl.static_range(W):
                b_y += b_x[i_r + i_w] * b_w[i_w][None, :]
            if bias is not None:
                b_y += b_bias[None, :]
            # backward preserves the baseline's rounded recomputed preactivation.
            b_y = b_y.to(x.dtype.element_ty).to(gl.float32)
            b_ys = 1. / (1. + gl.exp(-b_y))
            b_dyi = b_dyi * b_ys * (1 + b_y * (1 - b_ys))
        b_dy += (b_dyi,)
    b_db = gl.full((BT // BT_UNROLL, BD), 0, gl.float32, layout)
    for i_r in gl.static_range(BT_UNROLL):
        b_dx = gl.full((BT // BT_UNROLL, BD), 0, gl.float32, layout)
        b_dw_new = ()
        for i_w in gl.static_range(W):
            b_dyi = b_dy[i_r + i_w]
            b_dx += b_dyi * b_w[W - i_w - 1][None, :]
            b_dw_new += (b_dw[i_w] + b_dyi * b_x[i_r + W - 1],)
        b_dw = b_dw_new
        b_db += b_dy[i_r]
        gl.store(dx + (bos + o_t[:, None] + i_r) * D + o_d[None, :], b_dx,
                 (o_t[:, None] + i_r < T) & (o_d[None, :] < D))
    for i_w in gl.static_range(W):
        gl.store(dw_partial + i_tg * D * W + o_d * W + W - i_w - 1, gl.sum(b_dw[i_w], 0), o_d < D)
    if db_partial is not None:
        gl.store(db_partial + i_tg * D + o_d, gl.sum(b_db, 0), o_d < D)


def causal_conv1d_bwd(
    x, dy, dht, weight=None, bias=None, residual=None, initial_state=None, activation=None,
    cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, BT=64, layout_fallback=False,
):
    B, T, D = x.shape
    W = weight.shape[1]
    BD, num_warps = 32, 4
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    dw_partial = weight.new_empty(B * NT, D, W, dtype=torch.float32)
    db_partial = bias.new_empty(B * NT, D, dtype=torch.float32) if bias is not None else None
    causal_conv1d_bwd_kernel_gluon[(triton.cdiv(D, BD), NT, B)](
        x=x,
        weight=weight,
        bias=bias,
        dy=dy,
        dx=dx,
        dw_partial=dw_partial,
        db_partial=db_partial,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        stride_x_n=x.stride(0),
        stride_x_t=x.stride(1),
        stride_dy_n=dy.stride(0),
        stride_dy_t=dy.stride(1),
        stride_dy_d=dy.stride(2),
        D=D,
        W=W,
        BT=BT,
        BD=BD,
        BT_UNROLL=16,
        NUM_WARPS=num_warps,
        ACTIVATION=activation,
        num_warps=num_warps,
    )
    dw = dw_partial.sum(0).to(weight)
    db = db_partial.sum(0).to(bias) if db_partial is not None else None
    dr = dy if residual is not None else None
    return dx, dw, db, dr, None
