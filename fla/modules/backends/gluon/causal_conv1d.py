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
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    T,
    SXN: gl.constexpr,
    SXT: gl.constexpr,
    D: gl.constexpr,
    W: gl.constexpr,
    BT: gl.constexpr,
    BD: gl.constexpr,
    BC: gl.constexpr,
    ACTIVATION: gl.constexpr,
    NUM_SPLITS: gl.constexpr = 1,
):
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, BD // 32],
        threads_per_warp=[1, 32],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
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
        p_x = x + bos * SXT
    else:
        i_n = i_b
        bos = i_b * T
        p_x = x + i_b * SXN
    o_d = i_d * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    o_t = (i_t * NUM_SPLITS + i_split) * BT
    o_t += gl.arange(0, BT // BC, layout=gl.SliceLayout(1, layout)).to(gl.int64) * BC
    b_x = ()
    b_w = ()
    for i_w in gl.static_range(W):
        b_w += (gl.load(weight + o_d * W + i_w, mask=o_d < D, other=0).to(gl.float32),)
    # each row unrolls BC outputs from one register window; adjacent outputs reuse inputs.
    # with W=4:
    # y[t]   <- x[t-3] x[t-2] x[t-1] x[t]
    # y[t+1] <-        x[t-2] x[t-1] x[t] x[t+1]
    for i_x in gl.static_range(BC + W - 1):
        o_x = o_t + i_x - W + 1
        b_xi = gl.load(
            pointer=p_x + o_x[:, None] * SXT + o_d[None, :],
            mask=(o_x[:, None] >= 0) & (o_x[:, None] < T) & (o_d[None, :] < D),
            other=0,
        ).to(gl.float32)
        if initial_state is not None:
            b_xi += gl.load(
                pointer=initial_state + i_n * D * W + o_d[None, :] * W + (o_x[:, None] + W),
                mask=(o_x[:, None] < 0) & (o_x[:, None] >= 1 - W) & (o_d[None, :] < D),
                other=0,
            ).to(gl.float32)
        b_x += (b_xi,)
    if bias is not None:
        b_bias = gl.load(bias + o_d, mask=o_d < D, other=0).to(gl.float32)
    for i_r in gl.static_range(BC):
        b_y = gl.full((BT // BC, BD), 0, gl.float32, layout)
        for i_w in gl.static_range(W):
            b_y += b_x[i_r + i_w] * b_w[i_w][None, :]
        if bias is not None:
            b_y += b_bias[None, :]
        if ACTIVATION == 'silu' or ACTIVATION == 'swish':
            b_y *= 1. / (1. + gl.exp(-b_y))
        o_y = (bos + o_t[:, None] + i_r) * D + o_d[None, :]
        m_y = (o_t[:, None] + i_r < T) & (o_d[None, :] < D)
        if residual is not None:
            b_y += gl.load(residual + o_y, mask=m_y, other=0).to(gl.float32)
        gl.store(y + o_y, b_y, mask=m_y)


@gluon.jit(do_not_specialize=['T'])
def causal_conv1d_bwd_kernel(
    x,
    weight,
    bias,
    dy,
    dx,
    dw_partial,
    db_partial,
    cu_seqlens,
    chunk_indices,
    T,
    SXN: gl.constexpr,
    SXT: gl.constexpr,
    SYN: gl.constexpr,
    SYT: gl.constexpr,
    SYD: gl.constexpr,
    D: gl.constexpr,
    W: gl.constexpr,
    BT: gl.constexpr,
    BD: gl.constexpr,
    BC: gl.constexpr,
    ACTIVATION: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, BD // 32],
        threads_per_warp=[1, 32],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    i_d = gl.program_id(0).to(gl.int64)
    i_t = gl.program_id(1).to(gl.int64)
    i_b = gl.program_id(2).to(gl.int64)
    i_tg = i_b * gl.num_programs(1) + i_t
    if cu_seqlens is not None:
        i_n = gl.load(chunk_indices + 2 * i_t).to(gl.int64)
        i_t = gl.load(chunk_indices + 2 * i_t + 1).to(gl.int64)
        bos = gl.load(cu_seqlens + i_n).to(gl.int64)
        T = gl.load(cu_seqlens + i_n + 1).to(gl.int64) - bos
        p_x = x + bos * SXT
        p_dy = dy + bos * SYT
    else:
        bos = i_b * T
        p_x = x + i_b * SXN
        p_dy = dy + i_b * SYN
    o_d = i_d * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    # each row unrolls BC consecutive tokens within the BT-token tile.
    # rows (BT=64, BC=16): [0..15] [16..31] [32..47] [48..63]
    o_t = i_t * BT + gl.arange(0, BT // BC, layout=gl.SliceLayout(1, layout)).to(gl.int64) * BC
    b_w = ()
    b_x = ()
    b_dy = ()
    b_dw = ()
    for i_w in gl.static_range(W):
        b_w += (gl.load(weight + o_d * W + i_w, mask=o_d < D, other=0).to(gl.float32),)
        b_dw += (gl.full((BT // BC, BD), 0, gl.float32, layout),)
    if bias is not None:
        b_bias = gl.load(bias + o_d, mask=o_d < D, other=0).to(gl.float32)
    for i_x in gl.static_range(BC + 2 * W - 2):
        o_x = o_t + i_x - W + 1
        b_xi = gl.load(
            pointer=p_x + o_x[:, None] * SXT + o_d[None, :],
            mask=(o_x[:, None] >= 0) & (o_x[:, None] < T) & (o_d[None, :] < D),
            other=0,
        ).to(gl.float32)
        b_x += (b_xi,)
    for i_r in gl.static_range(BC + W - 1):
        b_dyi = gl.load(
            pointer=p_dy + (o_t[:, None] + i_r) * SYT + o_d[None, :] * SYD,
            mask=(o_t[:, None] + i_r < T) & (o_d[None, :] < D),
            other=0,
        ).to(gl.float32)
        if ACTIVATION == 'silu' or ACTIVATION == 'swish':
            b_y = gl.full((BT // BC, BD), 0, gl.float32, layout)
            for i_w in gl.static_range(W):
                b_y += b_x[i_r + i_w] * b_w[i_w][None, :]
            if bias is not None:
                b_y += b_bias[None, :]
            # backward preserves the baseline's rounded recomputed preactivation.
            b_y = b_y.to(x.dtype.element_ty).to(gl.float32)
            b_ys = 1. / (1. + gl.exp(-b_y))
            b_dyi = b_dyi * b_ys * (1 + b_y * (1 - b_ys))
        b_dy += (b_dyi,)
    b_db = gl.full((BT // BC, BD), 0, gl.float32, layout)
    for i_r in gl.static_range(BC):
        b_dx = gl.full((BT // BC, BD), 0, gl.float32, layout)
        b_dw_new = ()
        for i_w in gl.static_range(W):
            b_dyi = b_dy[i_r + i_w]
            b_dx += b_dyi * b_w[W - i_w - 1][None, :]
            b_dw_new += (b_dw[i_w] + b_dyi * b_x[i_r + W - 1],)
        b_dw = b_dw_new
        b_db += b_dy[i_r]
        gl.store(
            pointer=dx + (bos + o_t[:, None] + i_r) * D + o_d[None, :],
            value=b_dx,
            mask=(o_t[:, None] + i_r < T) & (o_d[None, :] < D),
        )
    for i_w in gl.static_range(W):
        gl.store(
            pointer=dw_partial + i_tg * D * W + o_d * W + W - i_w - 1,
            value=gl.sum(b_dw[i_w], axis=0),
            mask=o_d < D,
        )
    if db_partial is not None:
        gl.store(db_partial + i_tg * D + o_d, gl.sum(b_db, axis=0), mask=o_d < D)


@gluon.jit(do_not_specialize=['NP'])
def causal_conv1d_bwd_kernel_dwdb(
    dw_partial,
    db_partial,
    dw,
    db,
    NP,
    D: gl.constexpr,
    W: gl.constexpr,
    BD: gl.constexpr,
):
    BN: gl.constexpr = 128
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, BD // 32],
        threads_per_warp=[1, 32],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    i_d = gl.program_id(0).to(gl.int64)
    o_d = i_d * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    b_dw = gl.full((BD,), 0, gl.float32, gl.SliceLayout(0, layout))
    for i_n in range(0, NP, BN):
        o_n = i_n + gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
        b_partial = gl.load(
            pointer=dw_partial + o_n[:, None] * D * W + o_d[None, :],
            mask=(o_n[:, None] < NP) & (o_d[None, :] < D * W),
            other=0,
        )
        b_dw += gl.sum(b_partial, axis=0)
    gl.store(dw + o_d, b_dw.to(dw.dtype.element_ty), mask=o_d < D * W)
    if db_partial is not None:
        if i_d * BD < D:
            b_db = gl.full((BD,), 0, gl.float32, gl.SliceLayout(0, layout))
            for i_n in range(0, NP, BN):
                o_n = i_n + gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
                b_partial = gl.load(
                    pointer=db_partial + o_n[:, None] * D + o_d[None, :],
                    mask=(o_n[:, None] < NP) & (o_d[None, :] < D),
                    other=0,
                )
                b_db += gl.sum(b_partial, axis=0)
            gl.store(db + o_d, b_db.to(db.dtype.element_ty), mask=o_d < D)


@input_guard(no_guard_contiguous=['x'])
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    BT = chunk_size
    B, T, D = x.shape
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    use_small_tile = B * T * D <= 1048576
    BD, num_splits = (32, 2) if use_small_tile else (64, 1)
    causal_conv1d_fwd_kernel[(triton.cdiv(D, BD), NT * num_splits, B)](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        chunk_indices=chunk_indices,
        T=T,
        SXN=x.stride(0),
        SXT=x.stride(1),
        D=D,
        W=weight.shape[1],
        BT=BT // num_splits,
        BD=BD,
        BC=8,
        ACTIVATION=activation,
        NUM_SPLITS=num_splits,
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


def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, None]:
    BT = chunk_size
    B, T, D = x.shape
    W = weight.shape[1]
    BD = 32
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    dw_partial = weight.new_empty((B * NT, D, W), dtype=torch.float32)
    db_partial = bias.new_empty((B * NT, D), dtype=torch.float32) if bias is not None else None
    causal_conv1d_bwd_kernel[(triton.cdiv(D, BD), NT, B)](
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
        SXN=x.stride(0),
        SXT=x.stride(1),
        SYN=dy.stride(0),
        SYT=dy.stride(1),
        SYD=dy.stride(2),
        D=D,
        W=W,
        BT=BT,
        BD=BD,
        BC=16,
        ACTIVATION=activation,
        num_warps=4,
    )
    dw = weight.new_empty((D, W))
    db = bias.new_empty((D,)) if bias is not None else None
    causal_conv1d_bwd_kernel_dwdb[(triton.cdiv(D * W, BD),)](
        dw_partial=dw_partial,
        db_partial=db_partial,
        dw=dw,
        db=db,
        NP=B * NT,
        D=D,
        W=W,
        BD=BD,
        num_warps=4,
    )
    dr = dy if residual is not None else None
    return dx, dw, db, dr, None
