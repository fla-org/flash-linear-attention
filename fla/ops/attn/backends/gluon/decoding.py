# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy

from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.op import barrier, unflatten_program_id


@gluon.jit
def _copy_decode_tile(b_x_smem, p_x, m_x, USE_ASYNC_COPY: gl.constexpr):
    if USE_ASYNC_COPY:
        async_copy.async_copy_global_to_shared(b_x_smem, p_x, m_x)
    else:
        b_x_smem.store(gl.load(p_x, m_x, other=0.))


@gluon.jit
def attn_decoding_fwd_kernel_split(
    q,
    k,
    v,
    g_cumsum,
    sink_bias,
    cu_seqlens,
    o_partial,
    stats,
    o,
    H: gl.constexpr,
    HQ: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    scale: gl.constexpr,
    W: gl.constexpr,
    SPLITS: gl.constexpr,
    BS: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
    USE_G: gl.constexpr,
    USE_SINK_BIAS: gl.constexpr,
    USE_ASYNC_COPY: gl.constexpr,
):
    i_split, i_v, i_bh = unflatten_program_id(X=SPLITS, Y=gl.cdiv(V, BV))
    i_n, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // (HQ // H)
    bos = gl.load(cu_seqlens + i_n).to(gl.int64)
    eos = gl.load(cu_seqlens + i_n + 1).to(gl.int64)
    if W is not None:
        bos = gl.maximum(bos, eos - W)
    blocks = gl.cdiv(eos - bos, BS * SPLITS)
    i_first = bos + i_split * blocks * BS
    i_last = gl.minimum(i_first + blocks * BS, eos)
    # column-distributed warps keep the wide value reduction inside each warp.
    warps_per_cta: gl.constexpr = [1, NW] if BK >= 256 or BV >= 256 else [NW, 1]
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], warps_per_cta, [1, 0])
    o_t = gl.arange(0, BS, gl.SliceLayout(1, layout)).to(gl.int64)
    o_k = gl.arange(0, BK, gl.SliceLayout(0, layout)).to(gl.int64)
    o_v = i_v * BV + gl.arange(0, BV, gl.SliceLayout(0, layout)).to(gl.int64)
    b_q = gl.load(q + i_bh * K + o_k, o_k < K, other=0)
    b_q = (b_q * scale).to(b_q.dtype)
    layout_k: gl.constexpr = (
        gl.SwizzledSharedLayout(32, 1, 4, [1, 0]) if BK >= 128 else gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    layout_v: gl.constexpr = (
        gl.SwizzledSharedLayout(32, 1, 4, [1, 0]) if BV >= 128 else gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    b_k_smem = gl.allocate_shared_memory(b_q.dtype, [2, BS, BK], layout_k)
    b_v_smem = gl.allocate_shared_memory(b_q.dtype, [2, BS, BV], layout_v)
    p_k = k + ((i_first + o_t[:, None]) * H + i_h) * K + o_k[None, :]
    p_v = v + ((i_first + o_t[:, None]) * H + i_h) * V + o_v[None, :]
    if i_first < i_last:
        _copy_decode_tile(
            b_x_smem=b_k_smem.index(0),
            p_x=p_k,
            m_x=(i_first + o_t[:, None] < i_last) & (o_k[None, :] < K),
            USE_ASYNC_COPY=USE_ASYNC_COPY,
        )
        _copy_decode_tile(
            b_x_smem=b_v_smem.index(0),
            p_x=p_v,
            m_x=(i_first + o_t[:, None] < i_last) & (o_v[None, :] < V),
            USE_ASYNC_COPY=USE_ASYNC_COPY,
        )
        async_copy.commit_group()
    b_m = float('-inf')
    b_acc = 0.
    b_o = gl.full([BV], 0., gl.float32, gl.SliceLayout(0, layout))
    if USE_G:
        b_gq = gl.load(g_cumsum + (eos - 1) * HQ + i_hq, eos > bos, other=0).to(gl.float32)
    for i_s in range(gl.cdiv(gl.maximum(i_last - i_first, 0), BS)):
        i_slot = (i_s % 2).to(gl.int32)
        i_start = i_first + i_s * BS
        async_copy.wait_group(0)
        barrier()
        if i_start + BS < i_last:
            i_next = ((i_s + 1) % 2).to(gl.int32)
            o_t_next = i_start + BS + o_t[:, None]
            _copy_decode_tile(
                b_x_smem=b_k_smem.index(i_next),
                p_x=k + (o_t_next * H + i_h) * K + o_k[None, :],
                m_x=(o_t_next < i_last) & (o_k[None, :] < K),
                USE_ASYNC_COPY=USE_ASYNC_COPY,
            )
            _copy_decode_tile(
                b_x_smem=b_v_smem.index(i_next),
                p_x=v + (o_t_next * H + i_h) * V + o_v[None, :],
                m_x=(o_t_next < i_last) & (o_v[None, :] < V),
                USE_ASYNC_COPY=USE_ASYNC_COPY,
            )
            async_copy.commit_group()
        b_k = b_k_smem.index(i_slot).load(layout)
        b_v = b_v_smem.index(i_slot).load(layout)
        b_s = gl.sum(b_q[None, :] * b_k, 1).to(gl.float32)
        if USE_G:
            b_gk = gl.load(g_cumsum + (i_start + o_t) * HQ + i_hq, i_start + o_t < i_last, other=0).to(gl.float32)
            b_s += b_gq - b_gk
        b_s = gl.where(i_start + o_t < i_last, b_s, float('-inf')) * 1.4426950216
        b_m_new = gl.maximum(b_m, gl.max(b_s, 0))
        b_alpha = gl.exp2(b_m - b_m_new)
        b_p = gl.exp2(b_s - b_m_new)
        b_o = b_o * b_alpha + gl.sum(b_p[:, None] * b_v, 0)
        b_acc = b_acc * b_alpha + gl.sum(b_p, 0)
        b_m = b_m_new
    async_copy.wait_group(0)
    if SPLITS == 1:
        if USE_SINK_BIAS:
            b_m = gl.where(b_m == float('-inf'), 0., b_m)
            b_acc += gl.exp2(gl.load(sink_bias + i_hq).to(gl.float32) * 1.4426950216 - b_m)
        b_o /= gl.where(eos > bos, b_acc, 1.)
        gl.store(o + i_bh * V + o_v, b_o, o_v < V)
    else:
        gl.store(o_partial + (i_bh * SPLITS + i_split) * V + o_v, b_o, o_v < V)
        if i_v == 0:
            gl.store(stats + (i_bh * SPLITS + i_split) * 2, b_m)
            gl.store(stats + (i_bh * SPLITS + i_split) * 2 + 1, b_acc)


@gluon.jit
def attn_decoding_fwd_kernel_reduce(
    o_partial,
    stats,
    sink_bias,
    o,
    HQ: gl.constexpr,
    V: gl.constexpr,
    SPLITS: gl.constexpr,
    NS: gl.constexpr,
    BV: gl.constexpr,
    USE_SINK_BIAS: gl.constexpr,
):
    i_bh = gl.program_id(0).to(gl.int64)
    i_v = gl.program_id(1).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    o_split = gl.arange(0, NS, gl.SliceLayout(1, layout)).to(gl.int64)
    o_v = i_v * BV + gl.arange(0, BV, gl.SliceLayout(0, layout)).to(gl.int64)
    b_m_partial = gl.load(stats + (i_bh * SPLITS + o_split) * 2, o_split < SPLITS, other=float('-inf'))
    b_acc_partial = gl.load(stats + (i_bh * SPLITS + o_split) * 2 + 1, o_split < SPLITS, other=0.)
    b_m = gl.max(b_m_partial, 0)
    b_m = gl.where(b_m == float('-inf'), 0., b_m)
    b_alpha = gl.exp2(b_m_partial - b_m)
    b_acc = gl.sum(b_acc_partial * b_alpha, 0)
    if USE_SINK_BIAS:
        b_acc += gl.exp2(gl.load(sink_bias + i_bh % HQ).to(gl.float32) * 1.4426950216 - b_m)
    b_o_partial = gl.load(
        o_partial + (i_bh * SPLITS + o_split[:, None]) * V + o_v[None, :],
        (o_split[:, None] < SPLITS) & (o_v[None, :] < V),
        other=0.0,
    )
    b_o = gl.sum(b_o_partial * b_alpha[:, None], 0) / gl.where(b_acc > 0., b_acc, 1.)
    gl.store(o + i_bh * V + o_v, b_o, o_v < V)


def attn_decoding_one_step(
    q,
    k,
    v,
    g=None,
    scale=None,
    cu_seqlens=None,
    do_gate_scale=False,
    *,
    window_size=None,
    sink_bias=None,
):
    _, T, H, K = k.shape
    HQ, V = q.shape[2], v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    g_cumsum = chunk_global_cumsum(
        g,
        cu_seqlens=cu_seqlens,
        scale=scale if do_gate_scale else None,
        output_dtype=torch.float32,
    ) if g is not None else None
    N = len(cu_seqlens) - 1
    average = triton.cdiv(T, max(N, 1)) if window_size is None else min(window_size, triton.cdiv(T, max(N, 1)))
    splits = min(64, max(1, triton.cdiv(average, 256)))
    BK, BV = max(16, triton.next_power_of_2(K)), min(256, max(16, triton.next_power_of_2(V)))
    num_warps = 8 if max(K, V) >= 256 else 4
    o = torch.empty(*q.shape[:-1], V, dtype=v.dtype, device=q.device)
    o_partial = torch.empty(N * HQ, splits, V, dtype=torch.float32, device=q.device) if splits > 1 else o
    stats = torch.empty(N * HQ, splits, 2, dtype=torch.float32, device=q.device) if splits > 1 else o
    attn_decoding_fwd_kernel_split[(splits * N * HQ * triton.cdiv(V, BV),)](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        sink_bias=sink_bias,
        cu_seqlens=cu_seqlens,
        o_partial=o_partial,
        stats=stats,
        o=o,
        H=H,
        HQ=HQ,
        K=K,
        V=V,
        scale=scale,
        W=window_size,
        SPLITS=splits,
        BS=64 if max(K, V) <= 128 else 32,
        BK=BK,
        BV=BV,
        NW=num_warps,
        USE_G=g_cumsum is not None,
        USE_SINK_BIAS=sink_bias is not None,
        USE_ASYNC_COPY=K % 2 == 0 and V % 2 == 0 and k.data_ptr() % 16 == 0 and v.data_ptr() % 16 == 0,
        num_warps=num_warps,
    )
    if splits > 1:
        attn_decoding_fwd_kernel_reduce[(N * HQ, triton.cdiv(V, BV))](
            o_partial=o_partial,
            stats=stats,
            sink_bias=sink_bias,
            o=o,
            HQ=HQ,
            V=V,
            SPLITS=splits,
            NS=triton.next_power_of_2(splits),
            BV=BV,
            USE_SINK_BIAS=sink_bias is not None,
            num_warps=4,
        )
    return o
