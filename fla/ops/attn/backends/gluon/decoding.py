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
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

_thread_barrier = getattr(gl, 'thread_barrier', None) or getattr(gl, 'barrier', None)


@gluon.jit
def _copy_decode_tile(smem, ptr, mask, ASYNC: gl.constexpr):
    if ASYNC:
        cp.async_copy_global_to_shared(smem, ptr, mask)
    else:
        smem.store(gl.load(ptr, mask, other=0.))


@gluon.jit
def attn_decoding_fwd_kernel_split(
    Q, K, V, G, SINK, CU, PARTIAL, STATS, O,
    H: gl.constexpr, HQ: gl.constexpr, DK: gl.constexpr, DV: gl.constexpr,
    SCALE: gl.constexpr, W: gl.constexpr, SPLITS: gl.constexpr,
    BN: gl.constexpr, BK: gl.constexpr, BV: gl.constexpr, NW: gl.constexpr,
    USE_G: gl.constexpr, USE_SINK: gl.constexpr, ASYNC: gl.constexpr,
):
    split = gl.program_id(0).to(gl.int64)
    bh = gl.program_id(1).to(gl.int64)
    iv = gl.program_id(2).to(gl.int64)
    seq, hq = bh // HQ, bh % HQ
    hk = hq // (HQ // H)
    bos = gl.load(CU + seq).to(gl.int64)
    eos = gl.load(CU + seq + 1).to(gl.int64)
    if W is not None:
        bos = gl.maximum(bos, eos - W)
    blocks = gl.cdiv(eos - bos, BN * SPLITS)
    first = bos + split * blocks * BN
    last = gl.minimum(first + blocks * BN, eos)
    # column-distributed warps keep the wide value reduction inside each warp.
    warps: gl.constexpr = [1, NW] if BK >= 256 or BV >= 256 else [NW, 1]
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], warps, [1, 0])
    rows = gl.arange(0, BN, gl.SliceLayout(1, layout)).to(gl.int64)
    kd = gl.arange(0, BK, gl.SliceLayout(0, layout)).to(gl.int64)
    vd = iv * BV + gl.arange(0, BV, gl.SliceLayout(0, layout)).to(gl.int64)
    q = gl.load(Q + bh * DK + kd, kd < DK, other=0)
    q = (q * SCALE).to(q.dtype)
    kl: gl.constexpr = gl.SwizzledSharedLayout(32, 1, 4, [1, 0]) if BK >= 128 else gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
    vl: gl.constexpr = gl.SwizzledSharedLayout(32, 1, 4, [1, 0]) if BV >= 128 else gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
    ks = gl.allocate_shared_memory(q.dtype, [2, BN, BK], kl)
    vs = gl.allocate_shared_memory(q.dtype, [2, BN, BV], vl)
    kp = K + ((first + rows[:, None]) * H + hk) * DK + kd[None, :]
    vp = V + ((first + rows[:, None]) * H + hk) * DV + vd[None, :]
    if first < last:
        _copy_decode_tile(smem=ks.index(0), ptr=kp, mask=(first + rows[:, None] < last) & (kd[None, :] < DK), ASYNC=ASYNC)
        _copy_decode_tile(smem=vs.index(0), ptr=vp, mask=(first + rows[:, None] < last) & (vd[None, :] < DV), ASYNC=ASYNC)
        cp.commit_group()
    maximum = float('-inf')
    denom = 0.
    output = gl.full([BV], 0., gl.float32, gl.SliceLayout(0, layout))
    if USE_G:
        gq = gl.load(G + (eos - 1) * HQ + hq, eos > bos, other=0).to(gl.float32)
    for step in range(gl.cdiv(gl.maximum(last - first, 0), BN)):
        slot = (step % 2).to(gl.int32)
        start = first + step * BN
        cp.wait_group(0)
        _thread_barrier()
        if start + BN < last:
            nxt = ((step + 1) % 2).to(gl.int32)
            kr = start + BN + rows[:, None]
            _copy_decode_tile(
                smem=ks.index(nxt),
                ptr=K + (kr * H + hk) * DK + kd[None, :],
                mask=(kr < last) & (kd[None, :] < DK),
                ASYNC=ASYNC,
            )
            _copy_decode_tile(
                smem=vs.index(nxt),
                ptr=V + (kr * H + hk) * DV + vd[None, :],
                mask=(kr < last) & (vd[None, :] < DV),
                ASYNC=ASYNC,
            )
            cp.commit_group()
        keys = ks.index(slot).load(layout)
        values = vs.index(slot).load(layout)
        scores = gl.sum(q[None, :] * keys, 1)
        if USE_G:
            gk = gl.load(G + (start + rows) * HQ + hq, start + rows < last, other=0).to(gl.float32)
            scores += gq - gk
        scores = gl.where(start + rows < last, scores, float('-inf')) * 1.4426950216
        new_max = gl.maximum(maximum, gl.max(scores, 0))
        alpha = gl.exp2(maximum - new_max)
        p = gl.exp2(scores - new_max)
        output = output * alpha + gl.sum(p[:, None] * values, 0)
        denom = denom * alpha + gl.sum(p, 0)
        maximum = new_max
    cp.wait_group(0)
    if SPLITS == 1:
        if USE_SINK:
            maximum = gl.where(maximum == float('-inf'), 0., maximum)
            denom += gl.exp2(gl.load(SINK + hq).to(gl.float32) * 1.4426950216 - maximum)
        output /= gl.where(eos > bos, denom, 1.)
        gl.store(O + bh * DV + vd, output, vd < DV)
    else:
        gl.store(PARTIAL + (bh * SPLITS + split) * DV + vd, output, vd < DV)
        if iv == 0:
            gl.store(STATS + (bh * SPLITS + split) * 2, maximum)
            gl.store(STATS + (bh * SPLITS + split) * 2 + 1, denom)


@gluon.jit
def attn_decoding_fwd_kernel_reduce(
    PARTIAL, STATS, SINK, O, HQ: gl.constexpr, DV: gl.constexpr,
    SPLITS: gl.constexpr, BS: gl.constexpr, BV: gl.constexpr, USE_SINK: gl.constexpr,
):
    bh = gl.program_id(0).to(gl.int64)
    iv = gl.program_id(1).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    splits = gl.arange(0, BS, gl.SliceLayout(1, layout)).to(gl.int64)
    dims = iv * BV + gl.arange(0, BV, gl.SliceLayout(0, layout)).to(gl.int64)
    maxima = gl.load(STATS + (bh * SPLITS + splits) * 2, splits < SPLITS, other=float('-inf'))
    sums = gl.load(STATS + (bh * SPLITS + splits) * 2 + 1, splits < SPLITS, other=0.)
    maximum = gl.max(maxima, 0)
    maximum = gl.where(maximum == float('-inf'), 0., maximum)
    alpha = gl.exp2(maxima - maximum)
    denom = gl.sum(sums * alpha, 0)
    if USE_SINK:
        denom += gl.exp2(gl.load(SINK + bh % HQ).to(gl.float32) * 1.4426950216 - maximum)
    partial = gl.load(
        PARTIAL + (bh * SPLITS + splits[:, None]) * DV + dims[None, :],
        (splits[:, None] < SPLITS) & (dims[None, :] < DV),
        other=0.0,
    )
    output = gl.sum(partial * alpha[:, None], 0) / gl.where(denom > 0., denom, 1.)
    gl.store(O + bh * DV + dims, output, dims < DV)


def attn_decoding_gluon(
    q, k, v, g_cumsum, scale, cu_seqlens, window_size=None, sink_bias=None,
):
    _, t, h, dk = k.shape
    hq, dv = q.shape[2], v.shape[-1]
    n = len(cu_seqlens) - 1
    average = triton.cdiv(t, max(n, 1)) if window_size is None else min(window_size, triton.cdiv(t, max(n, 1)))
    splits = min(64, max(1, triton.cdiv(average, 256)))
    bk, bv = max(16, triton.next_power_of_2(dk)), min(256, max(16, triton.next_power_of_2(dv)))
    nw = 8 if max(dk, dv) >= 256 else 4
    out = torch.empty(*q.shape[:-1], dv, dtype=v.dtype, device=q.device)
    partial = torch.empty(n * hq, splits, dv, dtype=torch.float32, device=q.device) if splits > 1 else out
    stats = torch.empty(n * hq, splits, 2, dtype=torch.float32, device=q.device) if splits > 1 else out
    attn_decoding_fwd_kernel_split[(splits, n * hq, triton.cdiv(dv, bv))](
        Q=q,
        K=k,
        V=v,
        G=g_cumsum,
        SINK=sink_bias,
        CU=cu_seqlens,
        PARTIAL=partial,
        STATS=stats,
        O=out,
        H=h,
        HQ=hq,
        DK=dk,
        DV=dv,
        SCALE=scale,
        W=window_size,
        SPLITS=splits,
        BN=64 if max(dk, dv) <= 128 else 32,
        BK=bk,
        BV=bv,
        NW=nw,
        USE_G=g_cumsum is not None,
        USE_SINK=sink_bias is not None,
        ASYNC=dk % 2 == 0 and dv % 2 == 0 and (k.data_ptr() % 4 == 0) and (v.data_ptr() % 4 == 0),
        num_warps=nw,
    )
    if splits > 1:
        attn_decoding_fwd_kernel_reduce[(n * hq, triton.cdiv(dv, bv))](
            PARTIAL=partial,
            STATS=stats,
            SINK=sink_bias,
            O=out,
            HQ=hq,
            DV=dv,
            SPLITS=splits,
            BS=triton.next_power_of_2(splits),
            BV=bv,
            USE_SINK=sink_bias is not None,
            num_warps=4,
        )
    return out
