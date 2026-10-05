# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Register-window convolution with explicit channel layouts and fused backward recomputation."""

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
    D: gl.constexpr, W: gl.constexpr, BT: gl.constexpr, BD: gl.constexpr, R: gl.constexpr,
    V: gl.constexpr, NW: gl.constexpr, ACTIVATION: gl.constexpr, SPLIT: gl.constexpr = 1,
):
    # lanes own adjacent channels; each lane keeps R overlapping windows in registers.
    layout: gl.constexpr = gl.BlockedLayout([1, V], [1, 32], [NW, 1], [1, 0])
    i_d = gl.program_id(0).to(gl.int64)
    i_t = gl.program_id(1).to(gl.int64)
    i_n = gl.program_id(2).to(gl.int64)
    i_sub = i_t % SPLIT
    i_t //= SPLIT
    if cu_seqlens is not None:
        i_n = gl.load(chunk_indices + 2 * i_t).to(gl.int64)
        i_t = gl.load(chunk_indices + 2 * i_t + 1).to(gl.int64)
        bos = gl.load(cu_seqlens + i_n).to(gl.int64)
        T = gl.load(cu_seqlens + i_n + 1).to(gl.int64) - bos
        x += bos * stride_x_t
    else:
        bos = i_n * T
        x += i_n * stride_x_n
    d = i_d * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    t = (i_t * SPLIT + i_sub) * BT + gl.arange(0, BT // R, layout=gl.SliceLayout(1, layout)).to(gl.int64) * R
    values = ()
    weights = ()
    for k in gl.static_range(W):
        weights += (gl.load(weight + d * W + k, d < D, other=0).to(gl.float32),)
    for k in gl.static_range(R + W - 1):
        tx = t + k - W + 1
        value = gl.load(x + tx[:, None] * stride_x_t + d[None, :],
                        (tx[:, None] >= 0) & (tx[:, None] < T) & (d[None, :] < D), other=0).to(gl.float32)
        if initial_state is not None:
            value += gl.load(initial_state + i_n * D * W + d[None, :] * W + (tx[:, None] + W),
                             (tx[:, None] < 0) & (tx[:, None] >= 1 - W) & (d[None, :] < D), other=0).to(gl.float32)
        values += (value,)
    if bias is not None:
        b = gl.load(bias + d, d < D, other=0).to(gl.float32)
    for r in gl.static_range(R):
        acc = gl.full((BT // R, BD), 0, gl.float32, layout)
        for k in gl.static_range(W):
            acc += values[r + k] * weights[k][None, :]
        if bias is not None:
            acc += b[None, :]
        if ACTIVATION == 'silu' or ACTIVATION == 'swish':
            acc *= 1. / (1. + gl.exp(-acc))
        offset = (bos + t[:, None] + r) * D + d[None, :]
        mask = (t[:, None] + r < T) & (d[None, :] < D)
        if residual is not None:
            acc += gl.load(residual + offset, mask, other=0).to(gl.float32)
        gl.store(y + offset, acc, mask)


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
    small = B * T * D <= 262144
    BD, split = (32, 2) if small else (64, 1)
    causal_conv1d_fwd_kernel_gluon[(triton.cdiv(D, BD), NT * split, B)](
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
        BT=BT // split,
        BD=BD,
        R=8,
        V=BD // 32,
        NW=4,
        ACTIVATION=activation,
        SPLIT=split,
        num_warps=4,
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
    x, weight, bias, dy, dx, dw, db, cu_seqlens, chunk_indices,
    T, stride_x_n: gl.constexpr, stride_x_t: gl.constexpr,
    stride_dy_n: gl.constexpr, stride_dy_t: gl.constexpr, stride_dy_d: gl.constexpr,
    D: gl.constexpr, W: gl.constexpr, BT: gl.constexpr, BD: gl.constexpr, R: gl.constexpr,
    V: gl.constexpr, NW: gl.constexpr, ACTIVATION: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, V], [1, 32], [NW, 1], [1, 0])
    i_d = gl.program_id(0).to(gl.int64)
    i_t = gl.program_id(1).to(gl.int64)
    i_n = gl.program_id(2).to(gl.int64)
    i_partial = i_n * gl.num_programs(1) + i_t
    if cu_seqlens is not None:
        i_n = gl.load(chunk_indices + 2 * i_t).to(gl.int64)
        i_t = gl.load(chunk_indices + 2 * i_t + 1).to(gl.int64)
        bos = gl.load(cu_seqlens + i_n).to(gl.int64)
        T = gl.load(cu_seqlens + i_n + 1).to(gl.int64) - bos
        x += bos * stride_x_t
        dy += bos * stride_dy_t
    else:
        bos = i_n * T
        x += i_n * stride_x_n
        dy += i_n * stride_dy_n
    d = i_d * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    t = i_t * BT + gl.arange(0, BT // R, layout=gl.SliceLayout(1, layout)).to(gl.int64) * R
    weights = ()
    values = ()
    grads = ()
    dweights = ()
    for k in gl.static_range(W):
        weights += (gl.load(weight + d * W + k, d < D, other=0).to(gl.float32),)
        dweights += (gl.full((BT // R, BD), 0, gl.float32, layout),)
    if bias is not None:
        b = gl.load(bias + d, d < D, other=0).to(gl.float32)
    for k in gl.static_range(R + 2 * W - 2):
        tx = t + k - W + 1
        values += (gl.load(x + tx[:, None] * stride_x_t + d[None, :],
                           (tx[:, None] >= 0) & (tx[:, None] < T) & (d[None, :] < D), other=0).to(gl.float32),)
    for r in gl.static_range(R + W - 1):
        grad = gl.load(dy + (t[:, None] + r) * stride_dy_t + d[None, :] * stride_dy_d,
                       (t[:, None] + r < T) & (d[None, :] < D), other=0).to(gl.float32)
        if ACTIVATION == 'silu' or ACTIVATION == 'swish':
            pre = gl.full((BT // R, BD), 0, gl.float32, layout)
            for k in gl.static_range(W):
                pre += values[r + k] * weights[k][None, :]
            if bias is not None:
                pre += b[None, :]
            # backward preserves the baseline's rounded recomputed preactivation.
            pre = pre.to(x.dtype.element_ty).to(gl.float32)
            sig = 1. / (1. + gl.exp(-pre))
            grad = grad * sig * (1 + pre * (1 - sig))
        grads += (grad,)
    dbias = gl.full((BT // R, BD), 0, gl.float32, layout)
    for r in gl.static_range(R):
        acc = gl.full((BT // R, BD), 0, gl.float32, layout)
        updated = ()
        for k in gl.static_range(W):
            grad = grads[r + k]
            acc += grad * weights[W - k - 1][None, :]
            updated += (dweights[k] + grad * values[r + W - 1],)
        dweights = updated
        dbias += grads[r]
        gl.store(dx + (bos + t[:, None] + r) * D + d[None, :], acc,
                 (t[:, None] + r < T) & (d[None, :] < D))
    for k in gl.static_range(W):
        gl.store(dw + i_partial * D * W + d * W + W - k - 1, gl.sum(dweights[k], 0), d < D)
    if db is not None:
        gl.store(db + i_partial * D + d, gl.sum(dbias, 0), d < D)


def causal_conv1d_bwd(
    x, dy, dht, weight=None, bias=None, residual=None, initial_state=None, activation=None,
    cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, BT=64, layout_fallback=False,
):
    B, T, D = x.shape
    W = weight.shape[1]
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    dw = weight.new_empty(B * NT, D, W, dtype=torch.float32)
    db = bias.new_empty(B * NT, D, dtype=torch.float32) if bias is not None else None
    causal_conv1d_bwd_kernel_gluon[(triton.cdiv(D, 32), NT, B)](
        x=x,
        weight=weight,
        bias=bias,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
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
        BD=32,
        R=16,
        V=1,
        NW=4,
        ACTIVATION=activation,
        num_warps=4,
    )
    return dx, dw.sum(0).to(weight), db.sum(0).to(bias) if db is not None else None, dy if residual is not None else None, None
