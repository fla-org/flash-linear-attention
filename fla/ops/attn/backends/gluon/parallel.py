# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# async MMA staging follows https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html

import functools
import inspect

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import barrier, unflatten_program_id
from fla.utils import IS_TMA_SUPPORTED, get_device_capability

WARP_SPECIALIZE_V2 = 'functions_and_args' in inspect.signature(gl.warp_specialize).parameters


@gluon.constexpr_function
def _acc_layout(M, N, USE_TCGEN05, NW):
    if USE_TCGEN05:
        # dependency hashing inspects inactive branches, so both API names need guarded lookup.
        legacy = getattr(blackwell, 'get_tmem_32x32b_reg_layout', None)
        if legacy is not None:
            return legacy(M, min(N, 256), [M, N], NW)
        current = getattr(blackwell, 'get_tmem_reg_layout', None)
        return current(gl.float32, [M, N], _tmem_layout(M=M, N=N), NW)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[min(NW, M // 16), max(1, NW * 16 // M)],
        instr_shape=[16, min(N // max(1, NW * 16 // M), 256), 16],
    )


@gluon.constexpr_function
def _tmem_layout(M, N):
    if hasattr(blackwell, 'get_tmem_32x32b_reg_layout'):
        return blackwell.TensorMemoryLayout((M, min(N, 256)), unpacked=True)
    return blackwell.TensorMemoryLayout((M, min(N, 256)), col_stride=1)


@gluon.constexpr_function
def _packed_layout(M, N):
    if hasattr(blackwell, 'get_tmem_32x32b_reg_layout'):
        return blackwell.TensorMemoryLayout((M, N), unpacked=False)
    return blackwell.TensorMemoryLayout((M, N), col_stride=1)


@gluon.jit
def _packed_operand(acc_s, dtype: gl.constexpr, BT: gl.constexpr, BS: gl.constexpr):
    # packed operands reuse accumulators after their fp32 values have been consumed.
    desc_p = acc_s.slice(0, BS // 2)
    layout: gl.constexpr = _packed_layout(BT, BS)
    if gl.constexpr(hasattr(blackwell.tensor_memory_descriptor, '_reinterpret')):
        return desc_p._reinterpret(dtype, [BT, BS], layout)
    else:
        return desc_p.reinterpret(dtype, [BT, BS], layout)


@gluon.jit
def _acc_alloc(M: gl.constexpr, N: gl.constexpr, USE_TCGEN05: gl.constexpr, NW: gl.constexpr):
    layout: gl.constexpr = _acc_layout(M=M, N=N, USE_TCGEN05=USE_TCGEN05, NW=NW)
    b_acc = gl.zeros([M, N], gl.float32, layout)
    if USE_TCGEN05:
        tmem_layout: gl.constexpr = _tmem_layout(M=M, N=N)
        acc = blackwell.allocate_tensor_memory(gl.float32, [M, N], tmem_layout)
        acc.store(b_acc)
    else:
        acc = b_acc
    return acc


@gluon.jit
def _acc_read(acc, M: gl.constexpr, N: gl.constexpr, USE_TCGEN05: gl.constexpr, NW: gl.constexpr):
    if USE_TCGEN05:
        b_acc = acc.load(_acc_layout(M=M, N=N, USE_TCGEN05=USE_TCGEN05, NW=NW))
    else:
        b_acc = acc
    return b_acc


@gluon.jit
def _mma(a, b, acc, bar, phase, USE_TCGEN05: gl.constexpr, USE_ACC: gl.constexpr = True):
    if USE_TCGEN05:
        blackwell.tcgen05_mma(a, b, acc, use_acc=USE_ACC)
        blackwell.tcgen05_commit(bar)
        mbarrier.wait(bar, phase)
        phase ^= 1
    else:
        acc = warpgroup_mma(a, b, acc, use_acc=USE_ACC, is_async=True)
        acc = warpgroup_mma_wait(0, deps=(acc,))
    return acc, phase


@gluon.jit
def _load_tile(
    x,
    desc_x,
    b_x_smem,
    bar,
    i_h,
    i_t,
    eos,
    H: gl.constexpr,
    D: gl.constexpr,
    USE_TMA: gl.constexpr,
    NW: gl.constexpr = 4,
):
    if USE_TMA:
        tma.async_copy_global_to_shared(desc_x, [i_h.to(gl.int32), i_t.to(gl.int32), 0], bar, b_x_smem)
    else:
        BT: gl.constexpr = b_x_smem.shape[1]
        BD: gl.constexpr = b_x_smem.shape[2]
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
        o_t = i_t + gl.arange(0, BT, gl.SliceLayout(1, layout)).to(gl.int64)
        o_d = gl.arange(0, BD, gl.SliceLayout(0, layout)).to(gl.int64)
        b_x = gl.load(
            x + (o_t[:, None] * H + i_h) * D + o_d[None, :],
            (o_t[:, None] < eos) & (o_d[None, :] < D),
            other=0,
        )
        b_x_smem.reshape([BT, BD]).store(b_x)


@gluon.jit(do_not_specialize=['T', 'NT'])
def parallel_attn_fwd_kernel_gluon(
    q,
    k,
    v,
    o,
    lse,
    g_cumsum,
    sink_bias,
    cu_seqlens,
    chunk_indices,
    desc_q,
    desc_k,
    desc_v,
    T,
    NT,
    H: gl.constexpr,
    HQ: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    scale: gl.constexpr,
    W: gl.constexpr,
    BT: gl.constexpr,
    BS: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    BT_INDEX: gl.constexpr,
    USE_TCGEN05: gl.constexpr,
    USE_TMA_QK: gl.constexpr,
    USE_TMA_V: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    USE_G: gl.constexpr,
    USE_SINK_BIAS: gl.constexpr,
    NW: gl.constexpr,
):
    i_v, i_t, i_bh = unflatten_program_id(X=gl.cdiv(V, BV), Y=NT)
    i_hq = i_bh % HQ
    i_h = i_hq // (HQ // H)
    if IS_VARLEN:
        i_n = gl.load(chunk_indices + (i_t // (BT_INDEX // BT)) * 2).to(gl.int64)
        i_block = gl.load(chunk_indices + (i_t // (BT_INDEX // BT)) * 2 + 1).to(gl.int64)
        bos = gl.load(cu_seqlens + i_n).to(gl.int64)
        eos = gl.load(cu_seqlens + i_n + 1).to(gl.int64)
        i_start = bos + i_block * BT_INDEX + (i_t % (BT_INDEX // BT)) * BT
    else:
        bos = (i_bh // HQ) * T
        eos = bos + T
        i_start = bos + i_t * BT
    if i_start >= eos:
        return

    dtype: gl.constexpr = q.dtype.element_ty
    # three buffers keep the next prefetch disjoint from the previous in-flight PV.
    PIPELINED: gl.constexpr = not USE_TCGEN05 and BK <= 128 and BV <= 128
    NUM_BUFFERS: gl.constexpr = 3 if PIPELINED else 2
    layout_q_smem: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BT, BK], dtype)
    layout_k_smem: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BS, BK], dtype)
    layout_v_smem: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BS, BV], dtype)
    layout_p_smem: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BT, BS], dtype)
    b_q_smem = gl.allocate_shared_memory(dtype, [1, BT, BK], layout_q_smem)
    b_k_smem = gl.allocate_shared_memory(dtype, [NUM_BUFFERS, 1, BS, BK], layout_k_smem)
    b_v_smem = gl.allocate_shared_memory(dtype, [NUM_BUFFERS, 1, BS, BV], layout_v_smem)
    if not PIPELINED:
        b_p_smem = gl.allocate_shared_memory(dtype, [BT, BS], layout_p_smem)
    bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS + 2, 1], mbarrier.MBarrierLayout())
    for i_bar in gl.static_range(NUM_BUFFERS + 2):
        mbarrier.init(bars.index(i_bar), count=1)
    if USE_TMA_QK:
        mbarrier.expect(bars.index(NUM_BUFFERS), BT * BK * 2)
    _load_tile(
        x=q,
        desc_x=desc_q,
        b_x_smem=b_q_smem,
        bar=bars.index(NUM_BUFFERS),
        i_h=i_hq,
        i_t=i_start,
        eos=eos,
        H=HQ,
        D=K,
        USE_TMA=USE_TMA_QK,
        NW=NW,
    )
    if USE_TMA_QK:
        mbarrier.wait(bars.index(NUM_BUFFERS), 0)
    else:
        fence_async_shared()

    acc_s = _acc_alloc(M=BT, N=BS, USE_TCGEN05=USE_TCGEN05, NW=NW)
    acc_o = _acc_alloc(M=BT, N=BV, USE_TCGEN05=USE_TCGEN05, NW=NW)
    if PIPELINED:
        acc_o = warpgroup_mma_init(acc_o)
    layout_s: gl.constexpr = _acc_layout(M=BT, N=BS, USE_TCGEN05=USE_TCGEN05, NW=NW)
    layout_o: gl.constexpr = _acc_layout(M=BT, N=BV, USE_TCGEN05=USE_TCGEN05, NW=NW)
    o_t = i_start + gl.arange(0, BT, gl.SliceLayout(1, layout_s)).to(gl.int64)
    o_k = gl.arange(0, BS, gl.SliceLayout(0, layout_s)).to(gl.int64)
    b_m = gl.full([BT], float('-inf'), gl.float32, gl.SliceLayout(1, layout_s))
    b_acc = gl.full([BT], 0, gl.float32, gl.SliceLayout(1, layout_s))
    if USE_G:
        b_gq = gl.load(g_cumsum + o_t * HQ + i_hq, o_t < eos, other=0)
    i_first = bos
    if W is not None:
        i_first = bos + gl.maximum((i_start - bos - W + 1) // BS, 0) * BS
    i_last = gl.minimum(i_start + BT, eos)
    phase = 0
    for i_s in range(gl.cdiv(gl.maximum(i_last - i_first, 0), BS)):
        i_slot = (i_s % NUM_BUFFERS).to(gl.int32)
        i_key = i_first + i_s * BS
        if i_s == 0:
            if USE_TMA_QK or USE_TMA_V:
                mbarrier.expect(bars.index(i_slot), (BS * BK * 2 if USE_TMA_QK else 0) + (BS * BV * 2 if USE_TMA_V else 0))
            _load_tile(
                x=k,
                desc_x=desc_k,
                b_x_smem=b_k_smem.index(i_slot),
                bar=bars.index(i_slot),
                i_h=i_h,
                i_t=i_key,
                eos=eos,
                H=H,
                D=K,
                USE_TMA=USE_TMA_QK,
                NW=NW,
            )
            if USE_TMA_V:
                tma.async_copy_global_to_shared(
                    desc_v,
                    [i_h.to(gl.int32), i_key.to(gl.int32), (i_v * BV).to(gl.int32)],
                    bars.index(i_slot),
                    b_v_smem.index(i_slot),
                )
            else:
                layout_v: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
                o_vt = i_key + gl.arange(0, BS, gl.SliceLayout(1, layout_v)).to(gl.int64)
                o_vd = i_v * BV + gl.arange(0, BV, gl.SliceLayout(0, layout_v)).to(gl.int64)
                b_v = gl.load(
                    v + (o_vt[:, None] * H + i_h) * V + o_vd[None, :],
                    (o_vt[:, None] < eos) & (o_vd[None, :] < V),
                    other=0,
                )
                b_v_smem.index(i_slot).reshape([BS, BV]).store(b_v)
        if USE_TMA_QK or USE_TMA_V:
            mbarrier.wait(bars.index(i_slot), ((i_s // NUM_BUFFERS) % 2).to(gl.int32))
        if not USE_TMA_QK or not USE_TMA_V:
            fence_async_shared()
        if i_key + BS < i_last:
            i_next = ((i_s + 1) % NUM_BUFFERS).to(gl.int32)
            if USE_TMA_QK or USE_TMA_V:
                mbarrier.expect(bars.index(i_next), (BS * BK * 2 if USE_TMA_QK else 0) + (BS * BV * 2 if USE_TMA_V else 0))
            _load_tile(
                x=k,
                desc_x=desc_k,
                b_x_smem=b_k_smem.index(i_next),
                bar=bars.index(i_next),
                i_h=i_h,
                i_t=i_key + BS,
                eos=eos,
                H=H,
                D=K,
                USE_TMA=USE_TMA_QK,
                NW=NW,
            )
            if USE_TMA_V:
                tma.async_copy_global_to_shared(
                    desc_v,
                    [i_h.to(gl.int32), (i_key + BS).to(gl.int32), (i_v * BV).to(gl.int32)],
                    bars.index(i_next),
                    b_v_smem.index(i_next),
                )
            else:
                layout_v: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
                o_vt = i_key + BS + gl.arange(0, BS, gl.SliceLayout(1, layout_v)).to(gl.int64)
                o_vd = i_v * BV + gl.arange(0, BV, gl.SliceLayout(0, layout_v)).to(gl.int64)
                b_v = gl.load(
                    v + (o_vt[:, None] * H + i_h) * V + o_vd[None, :],
                    (o_vt[:, None] < eos) & (o_vd[None, :] < V),
                    other=0,
                )
                b_v_smem.index(i_next).reshape([BS, BV]).store(b_v)

        if PIPELINED:
            acc_s = warpgroup_mma(
                b_q_smem.reshape([BT, BK]),
                b_k_smem.index(i_slot).reshape([BS, BK]).permute((1, 0)),
                acc_s,
                use_acc=False,
                is_async=True,
            )
            acc_s, acc_o = warpgroup_mma_wait(0, deps=(acc_s, acc_o))
        else:
            acc_s, phase = _mma(
                a=b_q_smem.reshape([BT, BK]),
                b=b_k_smem.index(i_slot).reshape([BS, BK]).permute((1, 0)),
                acc=acc_s,
                bar=bars.index(NUM_BUFFERS + 1),
                phase=phase,
                USE_TCGEN05=USE_TCGEN05,
                USE_ACC=False,
            )
        b_s = _acc_read(acc=acc_s, M=BT, N=BS, USE_TCGEN05=USE_TCGEN05, NW=NW) * (scale * 1.4426950216)
        o_s = i_key + o_k
        if USE_G:
            b_gk = gl.load(g_cumsum + o_s * HQ + i_hq, o_s < eos, other=0)
            b_s += b_gq[:, None] - b_gk[None, :]
        if W is not None:
            m_s = (o_t[:, None] >= o_s[None, :]) & (o_s[None, :] < eos)
            m_s &= o_t[:, None] - o_s[None, :] < W
            b_s = gl.where(m_s, b_s, float('-inf'))
        elif USE_TCGEN05 or i_key + BS > i_start:
            m_s = (o_t[:, None] >= o_s[None, :]) & (o_s[None, :] < eos)
            b_s = gl.where(m_s, b_s, float('-inf'))
        b_m_new = gl.maximum(b_m, gl.max(b_s, 1))
        b_m_safe = gl.where(b_m_new == float('-inf'), 0., b_m_new)
        b_alpha = gl.exp2(b_m - b_m_safe)
        b_p = gl.exp2(b_s - b_m_safe[:, None])
        b_acc = b_acc * b_alpha + gl.sum(b_p, 1)
        b_m = b_m_new
        if USE_TCGEN05 and BT == 128 and BV == 256:
            # slicing keeps the wide output from occupying registers alongside the probability tile.
            b_p_smem.store(b_p.to(dtype))
            fence_async_shared()
            layout_c: gl.constexpr = _acc_layout(M=BT, N=64, USE_TCGEN05=USE_TCGEN05, NW=NW)
            for i_part in gl.static_range(BV // 64):
                acc_o_slice = acc_o.slice(i_part * 64, 64)
                b_value = acc_o_slice.load(layout_c)
                b_value *= gl.convert_layout(b_alpha, gl.SliceLayout(1, layout_c))[:, None]
                acc_o_slice.store(b_value)
        else:
            b_o_prev = _acc_read(acc=acc_o, M=BT, N=BV, USE_TCGEN05=USE_TCGEN05, NW=NW)
            b_o_prev *= gl.convert_layout(b_alpha, gl.SliceLayout(1, layout_o))[:, None]
            if USE_TCGEN05:
                acc_o.store(b_o_prev)
            else:
                acc_o = b_o_prev
            if PIPELINED:
                b_p_reg = gl.convert_layout(b_p.to(dtype), gl.DotOperandLayout(0, layout_o, 2))
            else:
                b_p_smem.store(b_p.to(dtype))
                fence_async_shared()
        if PIPELINED:
            acc_o = warpgroup_mma(b_p_reg, b_v_smem.index(i_slot).reshape([BS, BV]), acc_o, is_async=True)
        else:
            acc_o, phase = _mma(
                a=b_p_smem,
                b=b_v_smem.index(i_slot).reshape([BS, BV]),
                acc=acc_o,
                bar=bars.index(NUM_BUFFERS + 1),
                phase=phase,
                USE_TCGEN05=USE_TCGEN05,
            )

    if PIPELINED:
        acc_o = warpgroup_mma_wait(0, deps=(acc_o,))
    if USE_SINK_BIAS:
        b_sink = gl.load(sink_bias + i_hq)
        b_m = gl.where(b_m == float('-inf'), 0., b_m)
        b_acc += gl.exp2(b_sink - b_m)
    BC: gl.constexpr = 64 if USE_TCGEN05 and BT == 128 and BV == 256 else BV
    layout_c: gl.constexpr = _acc_layout(M=BT, N=BC, USE_TCGEN05=USE_TCGEN05, NW=NW)
    for i_part in gl.static_range(BV // BC):
        if USE_TCGEN05:
            b_o = acc_o.slice(i_part * BC, BC).load(layout_c)
        else:
            b_o = acc_o
        b_o /= gl.convert_layout(b_acc, gl.SliceLayout(1, layout_c))[:, None]
        o_ot = i_start + gl.arange(0, BT, gl.SliceLayout(1, layout_c)).to(gl.int64)
        o_ov = i_v * BV + i_part * BC + gl.arange(0, BC, gl.SliceLayout(0, layout_c)).to(gl.int64)
        gl.store(
            pointer=o + (o_ot[:, None] * HQ + i_hq) * V + o_ov[None, :],
            value=b_o,
            mask=(o_ot[:, None] < eos) & (o_ov[None, :] < V),
        )
    if i_v == 0:
        gl.store(lse + o_t * HQ + i_hq, b_m + gl.log2(b_acc), o_t < eos)
    for i_bar in gl.static_range(NUM_BUFFERS + 2):
        mbarrier.invalidate(bars.index(i_bar))


@gluon.jit
def _pipeline_load(args, descs, cfg: gl.constexpr):
    b_q_smem, b_k_smem, b_v_smem, acc_s, acc_o, bars, positions, pointers = args
    (
        bar_q_ready, bar_k_ready, bar_k_free, bar_v_ready, bar_v_free,
        bar_s_ready, bar_s_free, bar_p_ready, bar_o_free, bar_o_ready,
    ) = bars
    i_hq, i_h, i_start, eos, i_first, steps, i_v = positions
    desc_q, desc_k, desc_v = descs
    BT: gl.constexpr = cfg[0]
    BS: gl.constexpr = cfg[1]
    BK: gl.constexpr = cfg[2]
    BV: gl.constexpr = cfg[3]
    mbarrier.expect(bar_q_ready, BT * BK * 2)
    tma.async_copy_global_to_shared(desc_q, [i_hq.to(gl.int32), i_start.to(gl.int32), 0], bar_q_ready, b_q_smem)
    for i_s in range(steps):
        i_slot = i_s % 2
        phase = (i_s // 2) % 2
        i_t = (i_first + i_s * BS).to(gl.int32)
        mbarrier.wait(bar_k_free.index(i_slot), phase)
        mbarrier.expect(bar_k_ready.index(i_slot), BS * BK * 2)
        tma.async_copy_global_to_shared(desc_k, [i_h.to(gl.int32), i_t, 0], bar_k_ready.index(i_slot), b_k_smem.index(i_slot))
        mbarrier.wait(bar_v_free.index(i_slot), phase)
        mbarrier.expect(bar_v_ready.index(i_slot), BS * BV * 2)
        tma.async_copy_global_to_shared(
            desc_v,
            [i_h.to(gl.int32), i_t, (i_v * BV).to(gl.int32)],
            bar_v_ready.index(i_slot),
            b_v_smem.index(i_slot),
        )


@gluon.jit
def _pipeline_qk(
    b_q_smem,
    b_k_smem,
    acc_s,
    bar_k_ready,
    bar_k_free,
    bar_s_ready,
    bar_s_free,
    i_s,
    BT: gl.constexpr,
    BS: gl.constexpr,
    BK: gl.constexpr,
):
    i_slot = i_s % 2
    phase = (i_s // 2) % 2
    mbarrier.wait(bar_k_ready.index(i_slot), phase)
    mbarrier.wait(bar_s_free.index(i_slot), phase)
    blackwell.tcgen05_mma(
        a=b_q_smem.reshape([BT, BK]),
        b=b_k_smem.index(i_slot).reshape([BS, BK]).permute((1, 0)),
        acc=acc_s.index(i_slot),
        use_acc=False,
    )
    blackwell.tcgen05_commit(bar_s_ready.index(i_slot))
    blackwell.tcgen05_commit(bar_k_free.index(i_slot))


@gluon.jit
def _pipeline_mma(args, descs, cfg: gl.constexpr):
    b_q_smem, b_k_smem, b_v_smem, acc_s, acc_o, bars, positions, pointers = args
    (
        bar_q_ready, bar_k_ready, bar_k_free, bar_v_ready, bar_v_free,
        bar_s_ready, bar_s_free, bar_p_ready, bar_o_free, bar_o_ready,
    ) = bars
    i_hq, i_h, i_start, eos, i_first, steps, i_v = positions
    BT: gl.constexpr = cfg[0]
    BS: gl.constexpr = cfg[1]
    BK: gl.constexpr = cfg[2]
    BV: gl.constexpr = cfg[3]
    mbarrier.wait(bar_q_ready, 0)
    if steps > 0:
        _pipeline_qk(
            b_q_smem=b_q_smem,
            b_k_smem=b_k_smem,
            acc_s=acc_s,
            bar_k_ready=bar_k_ready,
            bar_k_free=bar_k_free,
            bar_s_ready=bar_s_ready,
            bar_s_free=bar_s_free,
            i_s=0,
            BT=BT,
            BS=BS,
            BK=BK,
        )
    for i_s in range(steps):
        # the next QK runs while the compute partition normalizes the current scores.
        if i_s + 1 < steps:
            _pipeline_qk(
                b_q_smem=b_q_smem,
                b_k_smem=b_k_smem,
                acc_s=acc_s,
                bar_k_ready=bar_k_ready,
                bar_k_free=bar_k_free,
                bar_s_ready=bar_s_ready,
                bar_s_free=bar_s_free,
                i_s=i_s + 1,
                BT=BT,
                BS=BS,
                BK=BK,
            )
        i_slot = i_s % 2
        phase = (i_s // 2) % 2
        mbarrier.wait(bar_p_ready.index(i_slot), phase)
        mbarrier.wait(bar_o_free, i_s % 2)
        mbarrier.wait(bar_v_ready.index(i_slot), phase)
        b_p = _packed_operand(acc_s=acc_s.index(i_slot), dtype=b_q_smem.dtype, BT=BT, BS=BS)
        blackwell.tcgen05_mma(a=b_p, b=b_v_smem.index(i_slot).reshape([BS, BV]), acc=acc_o, use_acc=i_s > 0)
        blackwell.tcgen05_commit(bar_o_ready)
        blackwell.tcgen05_commit(bar_s_free.index(i_slot))
        blackwell.tcgen05_commit(bar_v_free.index(i_slot))


@gluon.jit
def _pipeline_compute(args, descs, cfg: gl.constexpr):
    b_q_smem, b_k_smem, b_v_smem, acc_s, acc_o, bars, positions, pointers = args
    (
        bar_q_ready, bar_k_ready, bar_k_free, bar_v_ready, bar_v_free,
        bar_s_ready, bar_s_free, bar_p_ready, bar_o_free, bar_o_ready,
    ) = bars
    i_hq, i_h, i_start, eos, i_first, steps, i_v = positions
    o, lse, g_cumsum, sink_bias = pointers
    BT: gl.constexpr = cfg[0]
    BS: gl.constexpr = cfg[1]
    BV: gl.constexpr = cfg[3]
    HQ: gl.constexpr = cfg[4]
    V: gl.constexpr = cfg[5]
    scale: gl.constexpr = cfg[6]
    W: gl.constexpr = cfg[7]
    USE_G: gl.constexpr = cfg[8]
    USE_SINK_BIAS: gl.constexpr = cfg[9]
    dtype: gl.constexpr = b_q_smem.dtype
    layout_s: gl.constexpr = _acc_layout(M=BT, N=BS, USE_TCGEN05=True, NW=4)
    BC: gl.constexpr = 64 if BV >= 64 else BV
    layout_o: gl.constexpr = _acc_layout(M=BT, N=BC, USE_TCGEN05=True, NW=4)
    o_t = i_start + gl.arange(0, BT, gl.SliceLayout(1, layout_s)).to(gl.int64)
    o_s_local = gl.arange(0, BS, gl.SliceLayout(0, layout_s)).to(gl.int64)
    b_m = gl.full([BT], float('-inf'), gl.float32, gl.SliceLayout(1, layout_s))
    b_acc = gl.full([BT], 0., gl.float32, gl.SliceLayout(1, layout_s))
    if USE_G:
        b_gq = gl.load(g_cumsum + o_t * HQ + i_hq, o_t < eos, other=0)
    for i_s in range(steps):
        i_slot = i_s % 2
        phase = (i_s // 2) % 2
        mbarrier.wait(bar_s_ready.index(i_slot), phase)
        b_s = acc_s.index(i_slot).load(layout_s) * (scale * 1.4426950216)
        o_s = i_first + i_s * BS + o_s_local
        if USE_G:
            b_gk = gl.load(g_cumsum + o_s * HQ + i_hq, o_s < eos, other=0)
            b_s += b_gq[:, None] - b_gk[None, :]
        if W is not None:
            m_s = (o_t[:, None] >= o_s[None, :]) & (o_s[None, :] < eos)
            m_s &= o_t[:, None] - o_s[None, :] < W
            b_s = gl.where(m_s, b_s, float('-inf'))
        elif i_first + (i_s + 1) * BS > i_start:
            m_s = (o_t[:, None] >= o_s[None, :]) & (o_s[None, :] < eos)
            b_s = gl.where(m_s, b_s, float('-inf'))
        b_m_new = gl.maximum(b_m, gl.max(b_s, 1))
        b_m_safe = gl.where(b_m_new == float('-inf'), 0., b_m_new)
        b_alpha = gl.exp2(b_m - b_m_safe)
        b_p = gl.exp2(b_s - b_m_safe[:, None])
        b_m = b_m_new
        b_acc = b_acc * b_alpha + gl.sum(b_p, 1)
        _packed_operand(acc_s=acc_s.index(i_slot), dtype=dtype, BT=BT, BS=BS).store(b_p.to(dtype))
        mbarrier.arrive(bar_p_ready.index(i_slot), count=1)
        if i_s > 0:
            mbarrier.wait(bar_o_ready, (i_s - 1) % 2)
            for i_part in gl.static_range(BV // BC):
                acc_o_slice = acc_o.slice(i_part * BC, BC)
                b_o = acc_o_slice.load(layout_o)
                b_o *= gl.convert_layout(b_alpha, gl.SliceLayout(1, layout_o))[:, None]
                acc_o_slice.store(b_o)
        mbarrier.arrive(bar_o_free, count=1)
    if steps > 0:
        mbarrier.wait(bar_o_ready, (steps - 1) % 2)
    if USE_SINK_BIAS:
        b_sink = gl.load(sink_bias + i_hq)
        b_m = gl.where(b_m == float('-inf'), 0., b_m)
        b_acc += gl.exp2(b_sink - b_m)
    for i_part in gl.static_range(BV // BC):
        b_o = acc_o.slice(i_part * BC, BC).load(layout_o)
        b_o /= gl.convert_layout(b_acc, gl.SliceLayout(1, layout_o))[:, None]
        o_ot = i_start + gl.arange(0, BT, gl.SliceLayout(1, layout_o)).to(gl.int64)
        o_ov = i_v * BV + i_part * BC + gl.arange(0, BC, gl.SliceLayout(0, layout_o)).to(gl.int64)
        gl.store(o + (o_ot[:, None] * HQ + i_hq) * V + o_ov[None, :], b_o, (o_ot[:, None] < eos) & (o_ov[None, :] < V))
    if i_v == 0:
        gl.store(lse + o_t * HQ + i_hq, b_m + gl.log2(b_acc), o_t < eos)


@gluon.jit(do_not_specialize=['T', 'NT'])
def parallel_attn_fwd_kernel_pipeline(
    desc_q,
    desc_k,
    desc_v,
    o,
    lse,
    g_cumsum,
    sink_bias,
    cu_seqlens,
    chunk_indices,
    T,
    NT,
    H: gl.constexpr,
    HQ: gl.constexpr,
    V: gl.constexpr,
    scale: gl.constexpr,
    W: gl.constexpr,
    BT: gl.constexpr,
    BS: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    USE_G: gl.constexpr,
    USE_SINK_BIAS: gl.constexpr,
    WS_V2: gl.constexpr,
):
    i_v, i_t, i_bh = unflatten_program_id(X=gl.cdiv(V, BV), Y=NT)
    i_hq = i_bh % HQ
    i_h = i_hq // (HQ // H)
    if IS_VARLEN:
        i_n = gl.load(chunk_indices + i_t * 2).to(gl.int64)
        i_block = gl.load(chunk_indices + i_t * 2 + 1).to(gl.int64)
        bos = gl.load(cu_seqlens + i_n).to(gl.int64)
        eos = gl.load(cu_seqlens + i_n + 1).to(gl.int64)
        i_start = bos + i_block * BT
    else:
        bos = (i_bh // HQ) * T
        eos = bos + T
        i_start = bos + i_t * BT
    if i_start >= eos:
        return
    i_first = bos
    if W is not None:
        i_first = bos + gl.maximum((i_start - bos - W + 1) // BS, 0) * BS
    steps = gl.cdiv(gl.maximum(gl.minimum(i_start + BT, eos) - i_first, 0), BS).to(gl.int32)
    dtype: gl.constexpr = o.dtype.element_ty
    b_q_smem = gl.allocate_shared_memory(dtype, [1, BT, BK], desc_q.layout)
    b_k_smem = gl.allocate_shared_memory(dtype, [2, 1, BS, BK], desc_k.layout)
    b_v_smem = gl.allocate_shared_memory(dtype, [2, 1, BS, BV], desc_v.layout)
    acc_s = blackwell.allocate_tensor_memory(gl.float32, [2, BT, BS], _tmem_layout(M=BT, N=BS))
    acc_o = blackwell.allocate_tensor_memory(gl.float32, [BT, BV], _tmem_layout(M=BT, N=BV))
    acc_o.store(gl.zeros([BT, BV], gl.float32, _acc_layout(M=BT, N=BV, USE_TCGEN05=True, NW=4)))
    bar_q_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar_k_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    bar_k_free = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    bar_v_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    bar_v_free = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    bar_s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    bar_s_free = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    bar_p_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    bar_o_free = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar_o_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar_q_ready, count=1)
    mbarrier.init(bar_o_free, count=1)
    mbarrier.init(bar_o_ready, count=1)
    for j in gl.static_range(2):
        for bar in gl.static_range(7):
            bars = (bar_k_ready, bar_k_free, bar_v_ready, bar_v_free, bar_s_ready, bar_s_free, bar_p_ready)
            mbarrier.init(bars[bar].index(j), count=1)
            if bar % 2 == 1:
                mbarrier.arrive(bars[bar].index(j), count=1)
    bars = (bar_q_ready, bar_k_ready, bar_k_free, bar_v_ready, bar_v_free,
            bar_s_ready, bar_s_free, bar_p_ready, bar_o_free, bar_o_ready)
    positions = (i_hq, i_h, i_start, eos, i_first, steps, i_v)
    pointers = (o, lse, g_cumsum, sink_bias)
    args = (b_q_smem, b_k_smem, b_v_smem, acc_s, acc_o, bars, positions, pointers)
    descs = (desc_q, desc_k, desc_v)
    cfg: gl.constexpr = (BT, BS, BK, BV, HQ, V, scale, W, USE_G, USE_SINK_BIAS)
    if WS_V2:
        gl.warp_specialize(
            functions_and_args=[
                (_pipeline_compute, (args, descs, cfg)),
                (_pipeline_mma, (args, descs, cfg)),
                (_pipeline_load, (args, descs, cfg)),
            ],
            worker_num_warps=[1, 1],
            worker_num_regs=[24, 24],
        )
    else:
        gl.warp_specialize(
            default_args=(args, descs, cfg),
            default_partition=_pipeline_compute,
            worker_args=(args, descs, cfg),
            worker_partitions=[_pipeline_mma, _pipeline_load],
            worker_num_warps=[1, 1],
            worker_num_regs=[24, 24],
        )
    b_q_smem._keep_alive()
    b_k_smem._keep_alive()
    b_v_smem._keep_alive()
    for j in gl.static_range(2):
        for bar in gl.static_range(7):
            arrays = (bar_k_ready, bar_k_free, bar_v_ready, bar_v_free, bar_s_ready, bar_s_free, bar_p_ready)
            mbarrier.invalidate(arrays[bar].index(j))
    mbarrier.invalidate(bar_q_ready)
    mbarrier.invalidate(bar_o_free)
    mbarrier.invalidate(bar_o_ready)


@functools.lru_cache(maxsize=32)
def _descriptor_layout(BT, BD, dtype):
    return gl.NVMMASharedLayout.get_default_for([1, BT, BD], getattr(gl, str(dtype).split('.')[-1]))


def _descriptor(x, BT, BD):
    B, T, H, D = x.shape
    return TensorDescriptor(
        x,
        shape=[H, B * T, D],
        strides=[D, H * D, 1],
        block_shape=[1, BT, BD],
        layout=_descriptor_layout(BT=BT, BD=BD, dtype=x.dtype),
    )


def parallel_attn_fwd_gluon(q, k, v, g_cumsum, sink_bias, scale, window_size=None, cu_seqlens=None, chunk_indices=None):
    B, T, HQ, K = q.shape
    H, V = k.shape[2], v.shape[-1]
    BK, BV = max(64, triton.next_power_of_2(K)), min(256, max(32, triton.next_power_of_2(V)))
    use_tcgen05 = get_device_capability(q.device.index)[0] == 10
    num_warps = 4 if use_tcgen05 else 8
    BT, BS = (64, 32) if max(K, V) > 256 else (128, 64)
    if not use_tcgen05 and max(K, V) <= 128:
        BT, num_warps = 64, 4
    use_tma_qk = IS_TMA_SUPPORTED and K % 8 == 0 and q.data_ptr() % 16 == 0 and k.data_ptr() % 16 == 0
    use_tma_v = IS_TMA_SUPPORTED and V % 8 == 0 and v.data_ptr() % 16 == 0
    use_warp_specialization = use_tcgen05 and use_tma_qk and use_tma_v and BK <= 128 and BV <= 128
    BT_INDEX = 128 if chunk_indices is not None else BT
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT_INDEX)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices) * (BT_INDEX // BT)
    desc_q, desc_k = (_descriptor(x=q, BT=BT, BD=BK), _descriptor(x=k, BT=BS, BD=BK)) if use_tma_qk else (q, k)
    desc_v = _descriptor(x=v, BT=BS, BD=BV) if use_tma_v else v
    o = torch.empty(B, T, HQ, V, device=q.device, dtype=q.dtype)
    lse = torch.empty(B, T, HQ, device=q.device, dtype=torch.float32)
    if use_warp_specialization:
        parallel_attn_fwd_kernel_pipeline[(triton.cdiv(V, BV) * NT * B * HQ,)](
            desc_q=desc_q,
            desc_k=desc_k,
            desc_v=desc_v,
            o=o,
            lse=lse,
            g_cumsum=g_cumsum,
            sink_bias=sink_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            NT=NT,
            H=H,
            HQ=HQ,
            V=V,
            scale=scale,
            W=window_size,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            IS_VARLEN=cu_seqlens is not None,
            USE_G=g_cumsum is not None,
            USE_SINK_BIAS=sink_bias is not None,
            WS_V2=WARP_SPECIALIZE_V2,
            num_warps=4,
            maxnreg=128,
        )
        return o, lse
    parallel_attn_fwd_kernel_gluon[(triton.cdiv(V, BV) * NT * B * HQ,)](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        g_cumsum=g_cumsum,
        sink_bias=sink_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        desc_q=desc_q,
        desc_k=desc_k,
        desc_v=desc_v,
        T=T,
        NT=NT,
        H=H,
        HQ=HQ,
        K=K,
        V=V,
        scale=scale,
        W=window_size,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        BT_INDEX=BT_INDEX,
        USE_TCGEN05=use_tcgen05,
        USE_TMA_QK=use_tma_qk,
        USE_TMA_V=use_tma_v,
        IS_VARLEN=cu_seqlens is not None,
        USE_G=g_cumsum is not None,
        USE_SINK_BIAS=sink_bias is not None,
        NW=num_warps,
        num_warps=num_warps,
    )
    return o, lse


@gluon.jit
def _load_pair(
    a,
    desc_a,
    b_a_smem,
    b,
    desc_b,
    b_b_smem,
    bar,
    i_h,
    i_t,
    eos,
    H: gl.constexpr,
    D_A: gl.constexpr,
    D_B: gl.constexpr,
    USE_TMA_A: gl.constexpr,
    USE_TMA_B: gl.constexpr,
    NW: gl.constexpr = 4,
):
    if USE_TMA_A or USE_TMA_B:
        mbarrier.expect(
            bar,
            (b_a_smem.shape[1] * b_a_smem.shape[2] * 2 if USE_TMA_A else 0) +
            (b_b_smem.shape[1] * b_b_smem.shape[2] * 2 if USE_TMA_B else 0),
        )
    _load_tile(x=a, desc_x=desc_a, b_x_smem=b_a_smem, bar=bar, i_h=i_h, i_t=i_t, eos=eos, H=H, D=D_A, USE_TMA=USE_TMA_A, NW=NW)
    _load_tile(x=b, desc_x=desc_b, b_x_smem=b_b_smem, bar=bar, i_h=i_h, i_t=i_t, eos=eos, H=H, D=D_B, USE_TMA=USE_TMA_B, NW=NW)


@gluon.jit
def _mma_pair(a, b, c, d, acc_a, acc_b, bar, phase, USE_TCGEN05: gl.constexpr, USE_ACC: gl.constexpr = False):
    if USE_TCGEN05:
        blackwell.tcgen05_mma(a, b, acc_a, use_acc=USE_ACC)
        blackwell.tcgen05_mma(c, d, acc_b, use_acc=USE_ACC)
        blackwell.tcgen05_commit(bar)
        mbarrier.wait(bar, phase)
        phase ^= 1
    else:
        acc_a = warpgroup_mma(a, b, acc_a, use_acc=USE_ACC, is_async=True)
        acc_b = warpgroup_mma(c, d, acc_b, use_acc=USE_ACC, is_async=True)
        acc_a, acc_b = warpgroup_mma_wait(0, deps=(acc_a, acc_b))
    return acc_a, acc_b, phase


@gluon.jit(do_not_specialize=['T', 'NT'])
def parallel_attn_bwd_kernel_gluon(
    q,
    k,
    v,
    do,
    lse,
    delta,
    dq,
    dk,
    dv,
    g_cumsum,
    dg_cumsum,
    sink_bias,
    dsink_bias,
    cu_seqlens,
    chunk_indices,
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    T,
    NT,
    H: gl.constexpr,
    HQ: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    scale: gl.constexpr,
    W: gl.constexpr,
    BT: gl.constexpr,
    BS: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    BT_INDEX: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    USE_TCGEN05: gl.constexpr,
    USE_TMA_QK: gl.constexpr,
    USE_TMA_V: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    USE_G: gl.constexpr,
    USE_SINK_BIAS: gl.constexpr,
    GRAD: gl.constexpr,
    NW: gl.constexpr,
):
    IS_DQ: gl.constexpr = GRAD == 'dq'
    HAS_DS: gl.constexpr = GRAD != 'dv'
    HAS_DV: gl.constexpr = GRAD == 'dkv' or GRAD == 'dv'
    i_t, i_bh = unflatten_program_id(X=NT)
    i_hq = i_bh % HQ
    i_h = i_hq // (HQ // H)
    if IS_VARLEN:
        i_n = gl.load(chunk_indices + (i_t // (BT_INDEX // BT)) * 2).to(gl.int64)
        i_block = gl.load(chunk_indices + (i_t // (BT_INDEX // BT)) * 2 + 1).to(gl.int64)
        bos = gl.load(cu_seqlens + i_n).to(gl.int64)
        eos = gl.load(cu_seqlens + i_n + 1).to(gl.int64)
        i_start = bos + i_block * BT_INDEX + (i_t % (BT_INDEX // BT)) * BT
    else:
        bos = (i_bh // HQ) * T
        eos = bos + T
        i_start = bos + i_t * BT
    if i_start >= eos:
        return
    dtype: gl.constexpr = q.dtype.element_ty
    layout_q_smem: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BT, BK], dtype)
    layout_v_smem_resident: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BT, BV], dtype)
    layout_k_smem: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BS, BK], dtype)
    layout_do_smem: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BS, BV], dtype)
    layout_p_smem: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BT, BS], dtype)
    b_qk_smem = gl.allocate_shared_memory(dtype, [1, BT, BK], layout_q_smem)
    if HAS_DS:
        b_vo_smem = gl.allocate_shared_memory(dtype, [1, BT, BV], layout_v_smem_resident)
    b_qk_stream = gl.allocate_shared_memory(dtype, [NUM_BUFFERS, 1, BS, BK], layout_k_smem)
    b_vo_stream = gl.allocate_shared_memory(dtype, [NUM_BUFFERS, 1, BS, BV], layout_do_smem)
    b_ds_smem = gl.allocate_shared_memory(dtype, [BT, BS], layout_p_smem)
    b_p_smem = gl.allocate_shared_memory(dtype, [BT, BS], layout_p_smem)
    bars = gl.allocate_shared_memory(gl.int64, [4, 1], mbarrier.MBarrierLayout())
    for i_bar in gl.static_range(4):
        mbarrier.init(bars.index(i_bar), count=1)
    if IS_DQ:
        qk, vo = q, do
        qk_stream, vo_stream = k, v
        desc_qk, desc_vo = desc_q, desc_do
        desc_qk_stream, desc_vo_stream = desc_k, desc_v
        i_h_resident, H_RESIDENT = i_hq, HQ
        i_h_stream, H_STREAM = i_h, H
    else:
        qk, vo = k, v
        qk_stream, vo_stream = q, do
        desc_qk, desc_vo = desc_k, desc_v
        desc_qk_stream, desc_vo_stream = desc_q, desc_do
        i_h_resident, H_RESIDENT = i_h, H
        i_h_stream, H_STREAM = i_hq, HQ
    if USE_TMA_QK or (HAS_DS and USE_TMA_V):
        mbarrier.expect(bars.index(2), BT * BK * 2 * USE_TMA_QK + BT * BV * 2 * (HAS_DS and USE_TMA_V))
    _load_tile(
        x=qk,
        desc_x=desc_qk,
        b_x_smem=b_qk_smem,
        bar=bars.index(2),
        i_h=i_h_resident,
        i_t=i_start,
        eos=eos,
        H=H_RESIDENT,
        D=K,
        USE_TMA=USE_TMA_QK,
        NW=NW,
    )
    if HAS_DS:
        _load_tile(
            x=vo,
            desc_x=desc_vo,
            b_x_smem=b_vo_smem,
            bar=bars.index(2),
            i_h=i_h_resident,
            i_t=i_start,
            eos=eos,
            H=H_RESIDENT,
            D=V,
            USE_TMA=USE_TMA_V,
            NW=NW,
        )
    if USE_TMA_QK or (HAS_DS and USE_TMA_V):
        mbarrier.wait(bars.index(2), 0)
    if not USE_TMA_QK or (HAS_DS and not USE_TMA_V):
        fence_async_shared()

    layout_s: gl.constexpr = _acc_layout(M=BT, N=BS, USE_TCGEN05=USE_TCGEN05, NW=NW)
    o_t = i_start + gl.arange(0, BT, gl.SliceLayout(1, layout_s)).to(gl.int64)
    o_s_local = gl.arange(0, BS, gl.SliceLayout(0, layout_s)).to(gl.int64)
    REUSE_GRAD: gl.constexpr = USE_TCGEN05 and GRAD == 'dkv' and BK + BV > 512
    if HAS_DS:
        acc_dqk = _acc_alloc(M=BT, N=BK, USE_TCGEN05=USE_TCGEN05, NW=NW)
    if HAS_DV:
        acc_dv = _acc_alloc(M=BT, N=BV, USE_TCGEN05=USE_TCGEN05, NW=NW)
    if REUSE_GRAD:
        # preserve a small gradient slice while its TMEM holds the score and probability gradient.
        acc_scratch = acc_dqk if BK >= BV else acc_dv
        acc_s = acc_scratch.slice(0, BS)
        acc_dp = acc_scratch.slice(BS, BS)
    else:
        acc_s = _acc_alloc(M=BT, N=BS, USE_TCGEN05=USE_TCGEN05, NW=NW)
        if HAS_DS:
            acc_dp = _acc_alloc(M=BT, N=BS, USE_TCGEN05=USE_TCGEN05, NW=NW)
    if USE_G:
        b_gt = gl.load(g_cumsum + o_t * HQ + i_hq, o_t < eos, other=0)
        b_dg = gl.full([BT], 0, gl.float32, gl.SliceLayout(1, layout_s))
    if USE_SINK_BIAS and IS_DQ:
        b_sink_expectation = gl.full([BT], 0., gl.float32, gl.SliceLayout(1, layout_s))
    if IS_DQ:
        b_lse = gl.load(lse + o_t * HQ + i_hq, o_t < eos, other=0)
        b_delta = gl.load(delta + o_t * HQ + i_hq, o_t < eos, other=0)
        i_first = bos
        if W is not None:
            i_first = bos + gl.maximum((i_start - bos - W + 1) // BS, 0) * BS
        i_last = gl.minimum(i_start + BT, eos)
    else:
        i_first = i_start
        i_last = eos
        if W is not None:
            i_last = gl.minimum(i_last, i_start + BT + W - 1)
    phase = 0
    for i_s in range(gl.cdiv(gl.maximum(i_last - i_first, 0), BS)):
        i_slot = (i_s % NUM_BUFFERS).to(gl.int32)
        i_stream = i_first + i_s * BS
        if i_s == 0 or NUM_BUFFERS == 1:
            _load_pair(
                a=qk_stream,
                desc_a=desc_qk_stream,
                b_a_smem=b_qk_stream.index(i_slot),
                b=vo_stream,
                desc_b=desc_vo_stream,
                b_b_smem=b_vo_stream.index(i_slot),
                bar=bars.index(i_slot),
                i_h=i_h_stream,
                i_t=i_stream,
                eos=eos,
                H=H_STREAM,
                D_A=K,
                D_B=V,
                USE_TMA_A=USE_TMA_QK,
                USE_TMA_B=USE_TMA_V,
                NW=NW,
            )
        if USE_TMA_QK or USE_TMA_V:
            mbarrier.wait(bars.index(i_slot), ((i_s // NUM_BUFFERS) % 2).to(gl.int32))
        if not USE_TMA_QK or not USE_TMA_V:
            fence_async_shared()
        if NUM_BUFFERS > 1 and i_stream + BS < i_last:
            i_next = ((i_s + 1) % NUM_BUFFERS).to(gl.int32)
            _load_pair(
                a=qk_stream,
                desc_a=desc_qk_stream,
                b_a_smem=b_qk_stream.index(i_next),
                b=vo_stream,
                desc_b=desc_vo_stream,
                b_b_smem=b_vo_stream.index(i_next),
                bar=bars.index(i_next),
                i_h=i_h_stream,
                i_t=i_stream + BS,
                eos=eos,
                H=H_STREAM,
                D_A=K,
                D_B=V,
                USE_TMA_A=USE_TMA_QK,
                USE_TMA_B=USE_TMA_V,
                NW=NW,
            )
        if REUSE_GRAD:
            b_saved_grad = acc_scratch.slice(0, 2 * BS).load(_acc_layout(M=BT, N=2 * BS, USE_TCGEN05=USE_TCGEN05, NW=NW))
            # the MMA warp must not overwrite a slice while another warp is still saving it.
            barrier()
        if HAS_DS:
            acc_s, acc_dp, phase = _mma_pair(
                a=b_qk_smem.reshape([BT, BK]),
                b=b_qk_stream.index(i_slot).reshape([BS, BK]).permute((1, 0)),
                c=b_vo_smem.reshape([BT, BV]),
                d=b_vo_stream.index(i_slot).reshape([BS, BV]).permute((1, 0)),
                acc_a=acc_s,
                acc_b=acc_dp,
                bar=bars.index(3),
                phase=phase,
                USE_TCGEN05=USE_TCGEN05,
            )
        else:
            acc_s, phase = _mma(
                a=b_qk_smem.reshape([BT, BK]),
                b=b_qk_stream.index(i_slot).reshape([BS, BK]).permute((1, 0)),
                acc=acc_s,
                bar=bars.index(3),
                phase=phase,
                USE_TCGEN05=USE_TCGEN05,
                USE_ACC=False,
            )
        b_s = _acc_read(acc=acc_s, M=BT, N=BS, USE_TCGEN05=USE_TCGEN05, NW=NW) * (scale * 1.4426950216)
        o_s = i_stream + o_s_local
        if IS_DQ:
            use_mask = (i_stream + BS > i_start) | (i_start + BT > eos)
            b_norm = b_lse[:, None]
            b_delta_b = b_delta[:, None]
        else:
            use_mask = (i_start + BT > i_stream) | (i_stream + BS > eos)
            b_lse_col = gl.load(lse + o_s * HQ + i_hq, o_s < eos, other=0)
            b_delta_col = gl.load(delta + o_s * HQ + i_hq, o_s < eos, other=0)
            b_norm = b_lse_col[None, :]
            b_delta_b = b_delta_col[None, :]
        if USE_G:
            b_gs = gl.load(g_cumsum + o_s * HQ + i_hq, o_s < eos, other=0)
            if IS_DQ:
                b_s += b_gt[:, None] - b_gs[None, :]
            else:
                b_s += b_gs[None, :] - b_gt[:, None]
        b_p = gl.exp2(b_s - b_norm)
        if W is not None or use_mask:
            if IS_DQ:
                o_dist = o_t[:, None] - o_s[None, :]
            else:
                o_dist = o_s[None, :] - o_t[:, None]
            m_s = (o_dist >= 0) & (o_s[None, :] < eos) & (o_t[:, None] < eos)
            if W is not None:
                m_s &= o_dist < W
            b_p = gl.where(m_s, b_p, 0.)
        if HAS_DS:
            b_dp = _acc_read(acc=acc_dp, M=BT, N=BS, USE_TCGEN05=USE_TCGEN05, NW=NW)
            if USE_SINK_BIAS and IS_DQ:
                b_sink_expectation += gl.sum(b_p * b_dp, 1)
            b_ds = b_p * (b_dp - b_delta_b)
            if not USE_SINK_BIAS:
                # a single-key softmax has zero score gradient, independent of reduction roundoff.
                m_single = o_t[:, None] == bos if IS_DQ else o_s[None, :] == bos
                if W == 1:
                    m_single = gl.full([BT, BS], True, gl.int1, layout_s)
                b_ds = gl.where(m_single, 0., b_ds)
            if USE_G:
                b_dg += gl.sum(b_ds, 1) * (1 if IS_DQ else -1)
            b_ds_smem.store(b_ds.to(dtype))
        if HAS_DV:
            b_p_smem.store(b_p.to(dtype))
        fence_async_shared()
        if REUSE_GRAD:
            acc_scratch.slice(0, 2 * BS).store(b_saved_grad)
            # all warps must finish restoring the accumulator before the MMA warp reads it.
            barrier()
        if GRAD == 'dkv':
            acc_dqk, acc_dv, phase = _mma_pair(
                a=b_ds_smem,
                b=b_qk_stream.index(i_slot).reshape([BS, BK]),
                c=b_p_smem,
                d=b_vo_stream.index(i_slot).reshape([BS, BV]),
                acc_a=acc_dqk,
                acc_b=acc_dv,
                bar=bars.index(3),
                phase=phase,
                USE_TCGEN05=USE_TCGEN05,
                USE_ACC=True,
            )
        elif HAS_DS:
            acc_dqk, phase = _mma(
                a=b_ds_smem,
                b=b_qk_stream.index(i_slot).reshape([BS, BK]),
                acc=acc_dqk,
                bar=bars.index(3),
                phase=phase,
                USE_TCGEN05=USE_TCGEN05,
            )
        else:
            acc_dv, phase = _mma(
                a=b_p_smem,
                b=b_vo_stream.index(i_slot).reshape([BS, BV]),
                acc=acc_dv,
                bar=bars.index(3),
                phase=phase,
                USE_TCGEN05=USE_TCGEN05,
            )

    if USE_SINK_BIAS and IS_DQ:
        b_dsink = -gl.exp2(gl.load(sink_bias + i_hq) - b_lse) * b_sink_expectation
        gl.store(dsink_bias + o_t * HQ + i_hq, b_dsink, o_t < eos)
    if HAS_DS:
        b_dqk = _acc_read(acc=acc_dqk, M=BT, N=BK, USE_TCGEN05=USE_TCGEN05, NW=NW) * scale
        layout_dqk: gl.constexpr = _acc_layout(M=BT, N=BK, USE_TCGEN05=USE_TCGEN05, NW=NW)
        o_dqk_t = i_start + gl.arange(0, BT, gl.SliceLayout(1, layout_dqk)).to(gl.int64)
        o_dqk_d = gl.arange(0, BK, gl.SliceLayout(0, layout_dqk)).to(gl.int64)
        dqk = dq if IS_DQ else dk
        gl.store(
            dqk + (o_dqk_t[:, None] * HQ + i_hq) * K + o_dqk_d[None, :],
            b_dqk,
            (o_dqk_t[:, None] < eos) & (o_dqk_d[None, :] < K),
        )
        if USE_G:
            gl.store(dg_cumsum + o_t * HQ + i_hq, b_dg, o_t < eos)
    if HAS_DV:
        b_dv = _acc_read(acc=acc_dv, M=BT, N=BV, USE_TCGEN05=USE_TCGEN05, NW=NW)
        layout_dv: gl.constexpr = _acc_layout(M=BT, N=BV, USE_TCGEN05=USE_TCGEN05, NW=NW)
        o_vt = i_start + gl.arange(0, BT, gl.SliceLayout(1, layout_dv)).to(gl.int64)
        o_vd = gl.arange(0, BV, gl.SliceLayout(0, layout_dv)).to(gl.int64)
        gl.store(dv + (o_vt[:, None] * HQ + i_hq) * V + o_vd[None, :], b_dv, (o_vt[:, None] < eos) & (o_vd[None, :] < V))
    for i_bar in gl.static_range(4):
        mbarrier.invalidate(bars.index(i_bar))


def parallel_attn_bwd_gluon(
    q,
    k,
    v,
    o,
    g_cumsum,
    lse,
    do,
    sink_bias=None,
    scale=None,
    window_size=None,
    chunk_size=128,
    cu_seqlens=None,
    chunk_indices=None,
):
    from einops import reduce

    from fla.ops.attn.parallel import parallel_attn_bwd_preprocess

    B, T, HQ, K = q.shape
    H, V = k.shape[2], v.shape[-1]
    BK, BV = max(64, triton.next_power_of_2(K)), max(32, triton.next_power_of_2(V))
    use_tcgen05 = get_device_capability(q.device.index)[0] == 10
    num_warps = 4 if use_tcgen05 else 8
    if not use_tcgen05 and 64 < max(K, V) <= 128:
        num_warps = 4
    BT, BS, num_buffers = (64, 32, 1) if max(K, V) > 256 else (64, 64, 2)
    BT_INDEX = 128 if chunk_indices is not None else BT
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT_INDEX)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices) * (BT_INDEX // BT)
    delta = parallel_attn_bwd_preprocess(o=o, do=do)
    temp_dtype = q.dtype if H == HQ else torch.float32
    dq = torch.empty_like(q)
    dk_out = torch.empty(B, T, HQ, K, device=q.device, dtype=temp_dtype)
    dv_out = torch.empty(B, T, HQ, V, device=q.device, dtype=temp_dtype)
    dg_q = torch.empty_like(lse) if g_cumsum is not None else None
    dg_k = torch.empty_like(lse) if g_cumsum is not None else None
    dsink_rows = torch.empty_like(lse) if sink_bias is not None else None
    use_tma_qk = IS_TMA_SUPPORTED and K % 8 == 0 and q.data_ptr() % 16 == 0 and k.data_ptr() % 16 == 0
    use_tma_v = IS_TMA_SUPPORTED and V % 8 == 0 and v.data_ptr() % 16 == 0
    use_tma_v = use_tma_v and do.data_ptr() % 16 == 0
    gradients = ('dq', 'dkv') if use_tcgen05 or BK + BV <= 512 else ('dq', 'dk', 'dv')
    desc_q, desc_k = (_descriptor(x=q, BT=BT, BD=BK), _descriptor(x=k, BT=BS, BD=BK)) if use_tma_qk else (q, k)
    desc_v, desc_do = (_descriptor(x=v, BT=BS, BD=BV), _descriptor(x=do, BT=BT, BD=BV)) if use_tma_v else (v, do)
    dq_descriptors = desc_q, desc_k, desc_v, desc_do
    if BT != BS:
        desc_q, desc_k = (_descriptor(x=q, BT=BS, BD=BK), _descriptor(x=k, BT=BT, BD=BK)) if use_tma_qk else (q, k)
        desc_v, desc_do = (_descriptor(x=v, BT=BT, BD=BV), _descriptor(x=do, BT=BS, BD=BV)) if use_tma_v else (v, do)
    dkv_descriptors = desc_q, desc_k, desc_v, desc_do
    for grad in gradients:
        desc_q, desc_k, desc_v, desc_do = dq_descriptors if grad == 'dq' else dkv_descriptors
        parallel_attn_bwd_kernel_gluon[(NT * B * HQ,)](
            q=q,
            k=k,
            v=v,
            do=do,
            lse=lse,
            delta=delta,
            dq=dq,
            dk=dk_out,
            dv=dv_out,
            g_cumsum=g_cumsum,
            dg_cumsum=dg_q if grad == 'dq' else dg_k,
            sink_bias=sink_bias,
            dsink_bias=dsink_rows,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            desc_q=desc_q,
            desc_k=desc_k,
            desc_v=desc_v,
            desc_do=desc_do,
            T=T,
            NT=NT,
            H=H,
            HQ=HQ,
            K=K,
            V=V,
            scale=scale,
            W=window_size,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            BT_INDEX=BT_INDEX,
            NUM_BUFFERS=num_buffers,
            USE_TCGEN05=use_tcgen05,
            USE_TMA_QK=use_tma_qk,
            USE_TMA_V=use_tma_v,
            IS_VARLEN=cu_seqlens is not None,
            USE_G=g_cumsum is not None,
            USE_SINK_BIAS=sink_bias is not None,
            GRAD=grad,
            NW=num_warps,
            num_warps=num_warps,
        )
    if HQ != H:
        dk_out = reduce(dk_out, 'b t (h g) k -> b t h k', g=HQ // H, reduction='sum')
        dv_out = reduce(dv_out, 'b t (h g) v -> b t h v', g=HQ // H, reduction='sum')
    if g_cumsum is not None:
        dg_q.add_(dg_k)
    dsink = None if dsink_rows is None else dsink_rows.sum((0, 1))
    return dq, dk_out, dv_out, dg_q, dsink
