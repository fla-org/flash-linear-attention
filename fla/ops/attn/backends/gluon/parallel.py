# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# async MMA staging follows https://triton-lang.org/main/getting-started/tutorials/gluon/06-tcgen05.html

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
def _acc_layout(M, N, TCGEN, NW):
    if TCGEN:
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
def _packed_operand(score, dtype: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr):
    # packed operands reuse accumulators after their fp32 values have been consumed.
    ref = score.slice(0, BN // 2)
    layout: gl.constexpr = _packed_layout(BM, BN)
    if gl.constexpr(hasattr(blackwell.tensor_memory_descriptor, '_reinterpret')):
        return ref._reinterpret(dtype, [BM, BN], layout)
    else:
        return ref.reinterpret(dtype, [BM, BN], layout)


@gluon.jit
def _acc_alloc(M: gl.constexpr, N: gl.constexpr, TCGEN: gl.constexpr, NW: gl.constexpr):
    layout: gl.constexpr = _acc_layout(M=M, N=N, TCGEN=TCGEN, NW=NW)
    acc = gl.zeros([M, N], gl.float32, layout)
    if TCGEN:
        tmem_layout: gl.constexpr = _tmem_layout(M=M, N=N)
        result = blackwell.allocate_tensor_memory(gl.float32, [M, N], tmem_layout)
        result.store(acc)
    else:
        result = acc
    return result


@gluon.jit
def _acc_read(acc, M: gl.constexpr, N: gl.constexpr, TCGEN: gl.constexpr, NW: gl.constexpr):
    if TCGEN:
        value = acc.load(_acc_layout(M=M, N=N, TCGEN=TCGEN, NW=NW))
    else:
        value = acc
    return value


@gluon.jit
def _mma(a, b, acc, bar, phase, TCGEN: gl.constexpr, USE_ACC: gl.constexpr = True):
    if TCGEN:
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
    ptr,
    desc,
    smem,
    bar,
    head,
    row,
    end,
    H: gl.constexpr,
    D: gl.constexpr,
    TMA: gl.constexpr,
    NW: gl.constexpr = 4,
):
    if TMA:
        tma.async_copy_global_to_shared(desc, [head.to(gl.int32), row.to(gl.int32), 0], bar, smem)
    else:
        BM: gl.constexpr = smem.shape[1]
        BD: gl.constexpr = smem.shape[2]
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
        rows = row + gl.arange(0, BM, gl.SliceLayout(1, layout)).to(gl.int64)
        cols = gl.arange(0, BD, gl.SliceLayout(0, layout)).to(gl.int64)
        values = gl.load(
            ptr + (rows[:, None] * H + head) * D + cols[None, :],
            (rows[:, None] < end) & (cols[None, :] < D),
            other=0,
        )
        smem.reshape([BM, BD]).store(values)


@gluon.jit(do_not_specialize=['T'])
def parallel_attn_fwd_kernel_gluon(
    Q,
    K,
    V,
    O,
    LSE,
    GATE,
    SINK,
    CU,
    INDICES,
    Q_DESC,
    K_DESC,
    V_DESC,
    T,
    NT: gl.constexpr,
    H: gl.constexpr,
    HQ: gl.constexpr,
    DK: gl.constexpr,
    DV: gl.constexpr,
    SCALE: gl.constexpr,
    W: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    CHUNK: gl.constexpr,
    TCGEN: gl.constexpr,
    TMA_QK: gl.constexpr,
    TMA_V: gl.constexpr,
    VARLEN: gl.constexpr,
    USE_GATE: gl.constexpr,
    USE_SINK: gl.constexpr,
    NW: gl.constexpr,
):
    iv, it, bh = unflatten_program_id(X=gl.cdiv(DV, BV), Y=NT)
    hq = bh % HQ
    hk = hq // (HQ // H)
    if VARLEN:
        seq = gl.load(INDICES + (it // (CHUNK // BM)) * 2).to(gl.int64)
        block = gl.load(INDICES + (it // (CHUNK // BM)) * 2 + 1).to(gl.int64)
        bos = gl.load(CU + seq).to(gl.int64)
        end = gl.load(CU + seq + 1).to(gl.int64)
        start = bos + block * CHUNK + (it % (CHUNK // BM)) * BM
    else:
        bos = (bh // HQ) * T
        end = bos + T
        start = bos + it * BM
    if start >= end:
        return

    dtype: gl.constexpr = Q.dtype.element_ty
    # three buffers keep the next prefetch disjoint from the previous in-flight PV.
    PIPELINED: gl.constexpr = not TCGEN and BK <= 128 and BV <= 128
    BUFFERS: gl.constexpr = 3 if PIPELINED else 2
    q_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BM, BK], dtype)
    k_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BN, BK], dtype)
    v_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BN, BV], dtype)
    p_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, BN], dtype)
    qs = gl.allocate_shared_memory(dtype, [1, BM, BK], q_layout)
    ks = gl.allocate_shared_memory(dtype, [BUFFERS, 1, BN, BK], k_layout)
    vs = gl.allocate_shared_memory(dtype, [BUFFERS, 1, BN, BV], v_layout)
    if not PIPELINED:
        ps = gl.allocate_shared_memory(dtype, [BM, BN], p_layout)
    bars = gl.allocate_shared_memory(gl.int64, [BUFFERS + 2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(BUFFERS + 2):
        mbarrier.init(bars.index(i), count=1)
    if TMA_QK:
        mbarrier.expect(bars.index(BUFFERS), BM * BK * 2)
    _load_tile(
        ptr=Q,
        desc=Q_DESC,
        smem=qs,
        bar=bars.index(BUFFERS),
        head=hq,
        row=start,
        end=end,
        H=HQ,
        D=DK,
        TMA=TMA_QK,
        NW=NW,
    )
    if TMA_QK:
        mbarrier.wait(bars.index(BUFFERS), 0)
    else:
        fence_async_shared()

    score_acc = _acc_alloc(M=BM, N=BN, TCGEN=TCGEN, NW=NW)
    out_acc = _acc_alloc(M=BM, N=BV, TCGEN=TCGEN, NW=NW)
    if PIPELINED:
        out_acc = warpgroup_mma_init(out_acc)
    sl: gl.constexpr = _acc_layout(M=BM, N=BN, TCGEN=TCGEN, NW=NW)
    ol: gl.constexpr = _acc_layout(M=BM, N=BV, TCGEN=TCGEN, NW=NW)
    rows = start + gl.arange(0, BM, gl.SliceLayout(1, sl)).to(gl.int64)
    key_cols = gl.arange(0, BN, gl.SliceLayout(0, sl)).to(gl.int64)
    maximum = gl.full([BM], float('-inf'), gl.float32, gl.SliceLayout(1, sl))
    denom = gl.full([BM], 0, gl.float32, gl.SliceLayout(1, sl))
    if USE_GATE:
        gate_q = gl.load(GATE + rows * HQ + hq, rows < end, other=0)
    first = bos
    if W is not None:
        first = bos + gl.maximum((start - bos - W + 1) // BN, 0) * BN
    last = gl.minimum(start + BM, end)
    phase = 0
    for step in range(gl.cdiv(gl.maximum(last - first, 0), BN)):
        slot = (step % BUFFERS).to(gl.int32)
        key_start = first + step * BN
        if step == 0:
            if TMA_QK or TMA_V:
                mbarrier.expect(bars.index(slot), (BN * BK * 2 if TMA_QK else 0) + (BN * BV * 2 if TMA_V else 0))
            _load_tile(
                ptr=K,
                desc=K_DESC,
                smem=ks.index(slot),
                bar=bars.index(slot),
                head=hk,
                row=key_start,
                end=end,
                H=H,
                D=DK,
                TMA=TMA_QK,
                NW=NW,
            )
            if TMA_V:
                tma.async_copy_global_to_shared(
                    V_DESC,
                    [hk.to(gl.int32), key_start.to(gl.int32), (iv * BV).to(gl.int32)],
                    bars.index(slot),
                    vs.index(slot),
                )
            else:
                vl: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
                vr = key_start + gl.arange(0, BN, gl.SliceLayout(1, vl)).to(gl.int64)
                vc = iv * BV + gl.arange(0, BV, gl.SliceLayout(0, vl)).to(gl.int64)
                vv = gl.load(V + (vr[:, None] * H + hk) * DV + vc[None, :], (vr[:, None] < end) & (vc[None, :] < DV), other=0)
                vs.index(slot).reshape([BN, BV]).store(vv)
        if TMA_QK or TMA_V:
            mbarrier.wait(bars.index(slot), ((step // BUFFERS) % 2).to(gl.int32))
        if not TMA_QK or not TMA_V:
            fence_async_shared()
        if key_start + BN < last:
            nxt = ((step + 1) % BUFFERS).to(gl.int32)
            if TMA_QK or TMA_V:
                mbarrier.expect(bars.index(nxt), (BN * BK * 2 if TMA_QK else 0) + (BN * BV * 2 if TMA_V else 0))
            _load_tile(
                ptr=K,
                desc=K_DESC,
                smem=ks.index(nxt),
                bar=bars.index(nxt),
                head=hk,
                row=key_start + BN,
                end=end,
                H=H,
                D=DK,
                TMA=TMA_QK,
                NW=NW,
            )
            if TMA_V:
                tma.async_copy_global_to_shared(
                    V_DESC,
                    [hk.to(gl.int32), (key_start + BN).to(gl.int32), (iv * BV).to(gl.int32)],
                    bars.index(nxt),
                    vs.index(nxt),
                )
            else:
                vl: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
                vr = key_start + BN + gl.arange(0, BN, gl.SliceLayout(1, vl)).to(gl.int64)
                vc = iv * BV + gl.arange(0, BV, gl.SliceLayout(0, vl)).to(gl.int64)
                vv = gl.load(V + (vr[:, None] * H + hk) * DV + vc[None, :], (vr[:, None] < end) & (vc[None, :] < DV), other=0)
                vs.index(nxt).reshape([BN, BV]).store(vv)

        if PIPELINED:
            score_acc = warpgroup_mma(
                qs.reshape([BM, BK]),
                ks.index(slot).reshape([BN, BK]).permute((1, 0)),
                score_acc,
                use_acc=False,
                is_async=True,
            )
            score_acc, out_acc = warpgroup_mma_wait(0, deps=(score_acc, out_acc))
        else:
            score_acc, phase = _mma(
                a=qs.reshape([BM, BK]),
                b=ks.index(slot).reshape([BN, BK]).permute((1, 0)),
                acc=score_acc,
                bar=bars.index(BUFFERS + 1),
                phase=phase,
                TCGEN=TCGEN,
                USE_ACC=False,
            )
        scores = _acc_read(acc=score_acc, M=BM, N=BN, TCGEN=TCGEN, NW=NW) * (SCALE * 1.4426950216)
        cols = key_start + key_cols
        if USE_GATE:
            gate_k = gl.load(GATE + cols * HQ + hq, cols < end, other=0)
            scores += gate_q[:, None] - gate_k[None, :]
        if W is not None:
            mask = (rows[:, None] >= cols[None, :]) & (cols[None, :] < end)
            mask &= rows[:, None] - cols[None, :] < W
            scores = gl.where(mask, scores, float('-inf'))
        elif TCGEN or key_start + BN > start:
            mask = (rows[:, None] >= cols[None, :]) & (cols[None, :] < end)
            scores = gl.where(mask, scores, float('-inf'))
        new_max = gl.maximum(maximum, gl.max(scores, 1))
        finite_max = gl.where(new_max == float('-inf'), 0., new_max)
        alpha = gl.exp2(maximum - finite_max)
        prob = gl.exp2(scores - finite_max[:, None])
        denom = denom * alpha + gl.sum(prob, 1)
        maximum = new_max
        if TCGEN and BM == 128 and BV == 256:
            # slicing keeps the wide output from occupying registers alongside the probability tile.
            ps.store(prob.to(dtype))
            fence_async_shared()
            cl: gl.constexpr = _acc_layout(M=BM, N=64, TCGEN=TCGEN, NW=NW)
            for part in gl.static_range(BV // 64):
                tile = out_acc.slice(part * 64, 64)
                value = tile.load(cl)
                value *= gl.convert_layout(alpha, gl.SliceLayout(1, cl))[:, None]
                tile.store(value)
        else:
            old_out = _acc_read(acc=out_acc, M=BM, N=BV, TCGEN=TCGEN, NW=NW)
            old_out *= gl.convert_layout(alpha, gl.SliceLayout(1, ol))[:, None]
            if TCGEN:
                out_acc.store(old_out)
            else:
                out_acc = old_out
            if PIPELINED:
                p_reg = gl.convert_layout(prob.to(dtype), gl.DotOperandLayout(0, ol, 2))
            else:
                ps.store(prob.to(dtype))
                fence_async_shared()
        if PIPELINED:
            out_acc = warpgroup_mma(p_reg, vs.index(slot).reshape([BN, BV]), out_acc, is_async=True)
        else:
            out_acc, phase = _mma(
                a=ps,
                b=vs.index(slot).reshape([BN, BV]),
                acc=out_acc,
                bar=bars.index(BUFFERS + 1),
                phase=phase,
                TCGEN=TCGEN,
            )

    if PIPELINED:
        out_acc = warpgroup_mma_wait(0, deps=(out_acc,))
    if USE_SINK:
        sink = gl.load(SINK + hq)
        maximum = gl.where(maximum == float('-inf'), 0., maximum)
        denom += gl.exp2(sink - maximum)
    BC: gl.constexpr = 64 if TCGEN and BM == 128 and BV == 256 else BV
    cl: gl.constexpr = _acc_layout(M=BM, N=BC, TCGEN=TCGEN, NW=NW)
    for part in gl.static_range(BV // BC):
        if TCGEN:
            output = out_acc.slice(part * BC, BC).load(cl)
        else:
            output = out_acc
        output /= gl.convert_layout(denom, gl.SliceLayout(1, cl))[:, None]
        orows = start + gl.arange(0, BM, gl.SliceLayout(1, cl)).to(gl.int64)
        ocols = iv * BV + part * BC + gl.arange(0, BC, gl.SliceLayout(0, cl)).to(gl.int64)
        gl.store(
            pointer=O + (orows[:, None] * HQ + hq) * DV + ocols[None, :],
            value=output,
            mask=(orows[:, None] < end) & (ocols[None, :] < DV),
        )
    if iv == 0:
        gl.store(LSE + rows * HQ + hq, maximum + gl.log2(denom), rows < end)
    for i in gl.static_range(BUFFERS + 2):
        mbarrier.invalidate(bars.index(i))


@gluon.jit
def _pipeline_load(args, descs, cfg: gl.constexpr):
    qs, ks, vs, scores, output, bars, positions, pointers = args
    qb, kr, kf, vr, vf, sr, sf, pr, ore, ordy = bars
    hq, hk, start, end, first, steps, iv = positions
    q_desc, k_desc, v_desc = descs
    BM: gl.constexpr = cfg[0]
    BN: gl.constexpr = cfg[1]
    BK: gl.constexpr = cfg[2]
    BV: gl.constexpr = cfg[3]
    mbarrier.expect(qb, BM * BK * 2)
    tma.async_copy_global_to_shared(q_desc, [hq.to(gl.int32), start.to(gl.int32), 0], qb, qs)
    for i in range(steps):
        slot = i % 2
        phase = (i // 2) % 2
        row = (first + i * BN).to(gl.int32)
        mbarrier.wait(kf.index(slot), phase)
        mbarrier.expect(kr.index(slot), BN * BK * 2)
        tma.async_copy_global_to_shared(k_desc, [hk.to(gl.int32), row, 0], kr.index(slot), ks.index(slot))
        mbarrier.wait(vf.index(slot), phase)
        mbarrier.expect(vr.index(slot), BN * BV * 2)
        tma.async_copy_global_to_shared(v_desc, [hk.to(gl.int32), row, (iv * BV).to(gl.int32)], vr.index(slot), vs.index(slot))


@gluon.jit
def _pipeline_qk(qs, ks, scores, kr, kf, sr, sf, i, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    slot = i % 2
    phase = (i // 2) % 2
    mbarrier.wait(kr.index(slot), phase)
    mbarrier.wait(sf.index(slot), phase)
    blackwell.tcgen05_mma(
        a=qs.reshape([BM, BK]),
        b=ks.index(slot).reshape([BN, BK]).permute((1, 0)),
        acc=scores.index(slot),
        use_acc=False,
    )
    blackwell.tcgen05_commit(sr.index(slot))
    blackwell.tcgen05_commit(kf.index(slot))


@gluon.jit
def _pipeline_mma(args, descs, cfg: gl.constexpr):
    qs, ks, vs, scores, output, bars, positions, pointers = args
    qb, kr, kf, vr, vf, sr, sf, pr, ore, ordy = bars
    hq, hk, start, end, first, steps, iv = positions
    BM: gl.constexpr = cfg[0]
    BN: gl.constexpr = cfg[1]
    BK: gl.constexpr = cfg[2]
    BV: gl.constexpr = cfg[3]
    mbarrier.wait(qb, 0)
    if steps > 0:
        _pipeline_qk(qs=qs, ks=ks, scores=scores, kr=kr, kf=kf, sr=sr, sf=sf, i=0, BM=BM, BN=BN, BK=BK)
    for i in range(steps):
        # the next QK runs while the compute partition normalizes the current scores.
        if i + 1 < steps:
            _pipeline_qk(qs=qs, ks=ks, scores=scores, kr=kr, kf=kf, sr=sr, sf=sf, i=i + 1, BM=BM, BN=BN, BK=BK)
        slot = i % 2
        phase = (i // 2) % 2
        mbarrier.wait(pr.index(slot), phase)
        mbarrier.wait(ore, i % 2)
        mbarrier.wait(vr.index(slot), phase)
        prob = _packed_operand(score=scores.index(slot), dtype=qs.dtype, BM=BM, BN=BN)
        blackwell.tcgen05_mma(a=prob, b=vs.index(slot).reshape([BN, BV]), acc=output, use_acc=i > 0)
        blackwell.tcgen05_commit(ordy)
        blackwell.tcgen05_commit(sf.index(slot))
        blackwell.tcgen05_commit(vf.index(slot))


@gluon.jit
def _pipeline_compute(args, descs, cfg: gl.constexpr):
    qs, ks, vs, scores, output, bars, positions, pointers = args
    qb, kr, kf, vr, vf, sr, sf, pr, ore, ordy = bars
    hq, hk, start, end, first, steps, iv = positions
    O, LSE, GATE, SINK = pointers
    BM: gl.constexpr = cfg[0]
    BN: gl.constexpr = cfg[1]
    BV: gl.constexpr = cfg[3]
    HQ: gl.constexpr = cfg[4]
    DV: gl.constexpr = cfg[5]
    SCALE: gl.constexpr = cfg[6]
    W: gl.constexpr = cfg[7]
    USE_GATE: gl.constexpr = cfg[8]
    USE_SINK: gl.constexpr = cfg[9]
    dtype: gl.constexpr = qs.dtype
    sl: gl.constexpr = _acc_layout(M=BM, N=BN, TCGEN=True, NW=4)
    BC: gl.constexpr = 64 if BV >= 64 else BV
    ol: gl.constexpr = _acc_layout(M=BM, N=BC, TCGEN=True, NW=4)
    rows = start + gl.arange(0, BM, gl.SliceLayout(1, sl)).to(gl.int64)
    columns = gl.arange(0, BN, gl.SliceLayout(0, sl)).to(gl.int64)
    maximum = gl.full([BM], float('-inf'), gl.float32, gl.SliceLayout(1, sl))
    denom = gl.full([BM], 0., gl.float32, gl.SliceLayout(1, sl))
    if USE_GATE:
        gate_q = gl.load(GATE + rows * HQ + hq, rows < end, other=0)
    for i in range(steps):
        slot = i % 2
        phase = (i // 2) % 2
        mbarrier.wait(sr.index(slot), phase)
        score = scores.index(slot).load(sl) * (SCALE * 1.4426950216)
        cols = first + i * BN + columns
        if USE_GATE:
            gate_k = gl.load(GATE + cols * HQ + hq, cols < end, other=0)
            score += gate_q[:, None] - gate_k[None, :]
        if W is not None:
            mask = (rows[:, None] >= cols[None, :]) & (cols[None, :] < end)
            mask &= rows[:, None] - cols[None, :] < W
            score = gl.where(mask, score, float('-inf'))
        elif first + (i + 1) * BN > start:
            mask = (rows[:, None] >= cols[None, :]) & (cols[None, :] < end)
            score = gl.where(mask, score, float('-inf'))
        new_max = gl.maximum(maximum, gl.max(score, 1))
        finite_max = gl.where(new_max == float('-inf'), 0., new_max)
        alpha = gl.exp2(maximum - finite_max)
        prob = gl.exp2(score - finite_max[:, None])
        maximum = new_max
        denom = denom * alpha + gl.sum(prob, 1)
        _packed_operand(score=scores.index(slot), dtype=dtype, BM=BM, BN=BN).store(prob.to(dtype))
        mbarrier.arrive(pr.index(slot), count=1)
        if i > 0:
            mbarrier.wait(ordy, (i - 1) % 2)
            for part in gl.static_range(BV // BC):
                ref = output.slice(part * BC, BC)
                value = ref.load(ol)
                value *= gl.convert_layout(alpha, gl.SliceLayout(1, ol))[:, None]
                ref.store(value)
        mbarrier.arrive(ore, count=1)
    if steps > 0:
        mbarrier.wait(ordy, (steps - 1) % 2)
    if USE_SINK:
        sink = gl.load(SINK + hq)
        maximum = gl.where(maximum == float('-inf'), 0., maximum)
        denom += gl.exp2(sink - maximum)
    for part in gl.static_range(BV // BC):
        value = output.slice(part * BC, BC).load(ol)
        value /= gl.convert_layout(denom, gl.SliceLayout(1, ol))[:, None]
        rr = start + gl.arange(0, BM, gl.SliceLayout(1, ol)).to(gl.int64)
        cc = iv * BV + part * BC + gl.arange(0, BC, gl.SliceLayout(0, ol)).to(gl.int64)
        gl.store(O + (rr[:, None] * HQ + hq) * DV + cc[None, :], value, (rr[:, None] < end) & (cc[None, :] < DV))
    if iv == 0:
        gl.store(LSE + rows * HQ + hq, maximum + gl.log2(denom), rows < end)


@gluon.jit(do_not_specialize=['T'])
def parallel_attn_fwd_kernel_pipeline(
    Q_DESC,
    K_DESC,
    V_DESC,
    O,
    LSE,
    GATE,
    SINK,
    CU,
    INDICES,
    T,
    NT: gl.constexpr,
    H: gl.constexpr,
    HQ: gl.constexpr,
    DV: gl.constexpr,
    SCALE: gl.constexpr,
    W: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    VARLEN: gl.constexpr,
    USE_GATE: gl.constexpr,
    USE_SINK: gl.constexpr,
    WS_V2: gl.constexpr,
):
    iv, it, bh = unflatten_program_id(X=gl.cdiv(DV, BV), Y=NT)
    hq = bh % HQ
    hk = hq // (HQ // H)
    if VARLEN:
        seq = gl.load(INDICES + it * 2).to(gl.int64)
        block = gl.load(INDICES + it * 2 + 1).to(gl.int64)
        bos = gl.load(CU + seq).to(gl.int64)
        end = gl.load(CU + seq + 1).to(gl.int64)
        start = bos + block * BM
    else:
        bos = (bh // HQ) * T
        end = bos + T
        start = bos + it * BM
    if start >= end:
        return
    first = bos
    if W is not None:
        first = bos + gl.maximum((start - bos - W + 1) // BN, 0) * BN
    steps = gl.cdiv(gl.maximum(gl.minimum(start + BM, end) - first, 0), BN).to(gl.int32)
    dtype: gl.constexpr = O.dtype.element_ty
    qs = gl.allocate_shared_memory(dtype, [1, BM, BK], Q_DESC.layout)
    ks = gl.allocate_shared_memory(dtype, [2, 1, BN, BK], K_DESC.layout)
    vs = gl.allocate_shared_memory(dtype, [2, 1, BN, BV], V_DESC.layout)
    scores = blackwell.allocate_tensor_memory(gl.float32, [2, BM, BN], _tmem_layout(M=BM, N=BN))
    output = blackwell.allocate_tensor_memory(gl.float32, [BM, BV], _tmem_layout(M=BM, N=BV))
    output.store(gl.zeros([BM, BV], gl.float32, _acc_layout(M=BM, N=BV, TCGEN=True, NW=4)))
    qb = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    kr = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    kf = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    vr = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    vf = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    sr = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    sf = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    pr = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    ore = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    ordy = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(qb, count=1)
    mbarrier.init(ore, count=1)
    mbarrier.init(ordy, count=1)
    for j in gl.static_range(2):
        for bar in gl.static_range(7):
            bars = (kr, kf, vr, vf, sr, sf, pr)
            mbarrier.init(bars[bar].index(j), count=1)
            if bar % 2 == 1:
                mbarrier.arrive(bars[bar].index(j), count=1)
    bars = (qb, kr, kf, vr, vf, sr, sf, pr, ore, ordy)
    positions = (hq, hk, start, end, first, steps, iv)
    pointers = (O, LSE, GATE, SINK)
    args = (qs, ks, vs, scores, output, bars, positions, pointers)
    descs = (Q_DESC, K_DESC, V_DESC)
    cfg: gl.constexpr = (BM, BN, BK, BV, HQ, DV, SCALE, W, USE_GATE, USE_SINK)
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
    qs._keep_alive()
    ks._keep_alive()
    vs._keep_alive()
    for j in gl.static_range(2):
        for bar in gl.static_range(7):
            arrays = (kr, kf, vr, vf, sr, sf, pr)
            mbarrier.invalidate(arrays[bar].index(j))
    mbarrier.invalidate(qb)
    mbarrier.invalidate(ore)
    mbarrier.invalidate(ordy)


@functools.lru_cache(maxsize=32)
def _descriptor_layout(rows, dim, dtype):
    return gl.NVMMASharedLayout.get_default_for([1, rows, dim], getattr(gl, str(dtype).split('.')[-1]))


def _descriptor(x, rows, dim):
    b, t, h, d = x.shape
    return TensorDescriptor(
        x,
        shape=[h, b * t, d],
        strides=[d, h * d, 1],
        block_shape=[1, rows, dim],
        layout=_descriptor_layout(rows=rows, dim=dim, dtype=x.dtype),
    )


def parallel_attn_fwd_gluon(q, k, v, g_cumsum, sink_bias, scale, window_size=None, cu_seqlens=None, chunk_indices=None):
    b, t, hq, dk = q.shape
    h, dv = k.shape[2], v.shape[-1]
    bk, bv = max(64, triton.next_power_of_2(dk)), min(256, max(32, triton.next_power_of_2(dv)))
    tcgen = get_device_capability(q.device.index)[0] == 10
    nw = 4 if tcgen else 8
    bm, bn = (64, 32) if max(dk, dv) > 256 else (128, 64)
    if not tcgen and max(dk, dv) <= 128:
        bm, nw = 64, 4
    tma_qk = IS_TMA_SUPPORTED and dk % 8 == 0 and q.data_ptr() % 16 == 0 and k.data_ptr() % 16 == 0
    tma_v = IS_TMA_SUPPORTED and dv % 8 == 0 and v.data_ptr() % 16 == 0
    blackwell_pipeline = tcgen and tma_qk and tma_v and bk <= 128 and bv <= 128
    chunk = 128 if chunk_indices is not None else bm
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=chunk)
    nt = triton.cdiv(t, bm) if cu_seqlens is None else len(chunk_indices) * (chunk // bm)
    q_desc, k_desc = (_descriptor(x=q, rows=bm, dim=bk), _descriptor(x=k, rows=bn, dim=bk)) if tma_qk else (q, k)
    v_desc = _descriptor(x=v, rows=bn, dim=bv) if tma_v else v
    o = torch.empty(b, t, hq, dv, device=q.device, dtype=q.dtype)
    lse = torch.empty(b, t, hq, device=q.device, dtype=torch.float32)
    if blackwell_pipeline:
        parallel_attn_fwd_kernel_pipeline[(triton.cdiv(dv, bv) * nt * b * hq,)](
            Q_DESC=q_desc,
            K_DESC=k_desc,
            V_DESC=v_desc,
            O=o,
            LSE=lse,
            GATE=g_cumsum,
            SINK=sink_bias,
            CU=cu_seqlens,
            INDICES=chunk_indices,
            T=t,
            NT=nt,
            H=h,
            HQ=hq,
            DV=dv,
            SCALE=scale,
            W=window_size,
            BM=bm,
            BN=bn,
            BK=bk,
            BV=bv,
            VARLEN=cu_seqlens is not None,
            USE_GATE=g_cumsum is not None,
            USE_SINK=sink_bias is not None,
            WS_V2=WARP_SPECIALIZE_V2,
            num_warps=4,
            maxnreg=128,
        )
        return o, lse
    parallel_attn_fwd_kernel_gluon[(triton.cdiv(dv, bv) * nt * b * hq,)](
        Q=q,
        K=k,
        V=v,
        O=o,
        LSE=lse,
        GATE=g_cumsum,
        SINK=sink_bias,
        CU=cu_seqlens,
        INDICES=chunk_indices,
        Q_DESC=q_desc,
        K_DESC=k_desc,
        V_DESC=v_desc,
        T=t,
        NT=nt,
        H=h,
        HQ=hq,
        DK=dk,
        DV=dv,
        SCALE=scale,
        W=window_size,
        BM=bm,
        BN=bn,
        BK=bk,
        BV=bv,
        CHUNK=chunk,
        TCGEN=tcgen,
        TMA_QK=tma_qk,
        TMA_V=tma_v,
        VARLEN=cu_seqlens is not None,
        USE_GATE=g_cumsum is not None,
        USE_SINK=sink_bias is not None,
        NW=nw,
        num_warps=nw,
    )
    return o, lse


@gluon.jit
def _load_pair(
    A,
    AD,
    AS,
    B,
    BD,
    BS,
    bar,
    head,
    row,
    end,
    H: gl.constexpr,
    DA: gl.constexpr,
    DB: gl.constexpr,
    TMA_A: gl.constexpr,
    TMA_B: gl.constexpr,
    NW: gl.constexpr = 4,
):
    if TMA_A or TMA_B:
        mbarrier.expect(bar, (AS.shape[1] * AS.shape[2] * 2 if TMA_A else 0) + (BS.shape[1] * BS.shape[2] * 2 if TMA_B else 0))
    _load_tile(ptr=A, desc=AD, smem=AS, bar=bar, head=head, row=row, end=end, H=H, D=DA, TMA=TMA_A, NW=NW)
    _load_tile(ptr=B, desc=BD, smem=BS, bar=bar, head=head, row=row, end=end, H=H, D=DB, TMA=TMA_B, NW=NW)


@gluon.jit
def _mma_pair(a, b, c, d, acc_a, acc_b, bar, phase, TCGEN: gl.constexpr, USE_ACC: gl.constexpr = False):
    if TCGEN:
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


@gluon.jit(do_not_specialize=['T'])
def parallel_attn_bwd_kernel_gluon(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    DK_OUT,
    DV_OUT,
    GATE,
    DG,
    SINK,
    DSINK,
    CU,
    INDICES,
    Q_DESC,
    K_DESC,
    V_DESC,
    DO_DESC,
    T,
    NT: gl.constexpr,
    H: gl.constexpr,
    HQ: gl.constexpr,
    DK: gl.constexpr,
    DV: gl.constexpr,
    SCALE: gl.constexpr,
    W: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    CHUNK: gl.constexpr,
    BUFFERS: gl.constexpr,
    TCGEN: gl.constexpr,
    TMA_QK: gl.constexpr,
    TMA_V: gl.constexpr,
    VARLEN: gl.constexpr,
    USE_GATE: gl.constexpr,
    USE_SINK: gl.constexpr,
    GRAD: gl.constexpr,
    NW: gl.constexpr,
):
    IS_DQ: gl.constexpr = GRAD == 'dq'
    HAS_DS: gl.constexpr = GRAD != 'dv'
    HAS_DV: gl.constexpr = GRAD == 'dkv' or GRAD == 'dv'
    it, bh = unflatten_program_id(X=NT)
    hq = bh % HQ
    hk = hq // (HQ // H)
    if VARLEN:
        seq = gl.load(INDICES + (it // (CHUNK // BM)) * 2).to(gl.int64)
        block = gl.load(INDICES + (it // (CHUNK // BM)) * 2 + 1).to(gl.int64)
        bos = gl.load(CU + seq).to(gl.int64)
        end = gl.load(CU + seq + 1).to(gl.int64)
        start = bos + block * CHUNK + (it % (CHUNK // BM)) * BM
    else:
        bos = (bh // HQ) * T
        end = bos + T
        start = bos + it * BM
    if start >= end:
        return
    dtype: gl.constexpr = Q.dtype.element_ty
    ql: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BM, BK], dtype)
    vl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BM, BV], dtype)
    kl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BN, BK], dtype)
    dl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([1, BN, BV], dtype)
    pl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, BN], dtype)
    resident_a = gl.allocate_shared_memory(dtype, [1, BM, BK], ql)
    if HAS_DS:
        resident_b = gl.allocate_shared_memory(dtype, [1, BM, BV], vl)
    stream_a = gl.allocate_shared_memory(dtype, [BUFFERS, 1, BN, BK], kl)
    stream_b = gl.allocate_shared_memory(dtype, [BUFFERS, 1, BN, BV], dl)
    ds_shared = gl.allocate_shared_memory(dtype, [BM, BN], pl)
    p_shared = gl.allocate_shared_memory(dtype, [BM, BN], pl)
    bars = gl.allocate_shared_memory(gl.int64, [4, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(4):
        mbarrier.init(bars.index(i), count=1)
    if IS_DQ:
        resident_qk, resident_value = Q, DO
        stream_qk, stream_value = K, V
        resident_qk_desc, resident_value_desc = Q_DESC, DO_DESC
        stream_qk_desc, stream_value_desc = K_DESC, V_DESC
        resident_head, resident_heads = hq, HQ
        stream_head, stream_heads = hk, H
    else:
        resident_qk, resident_value = K, V
        stream_qk, stream_value = Q, DO
        resident_qk_desc, resident_value_desc = K_DESC, V_DESC
        stream_qk_desc, stream_value_desc = Q_DESC, DO_DESC
        resident_head, resident_heads = hk, H
        stream_head, stream_heads = hq, HQ
    if TMA_QK or (HAS_DS and TMA_V):
        mbarrier.expect(bars.index(2), BM * BK * 2 * TMA_QK + BM * BV * 2 * (HAS_DS and TMA_V))
    _load_tile(
        ptr=resident_qk,
        desc=resident_qk_desc,
        smem=resident_a,
        bar=bars.index(2),
        head=resident_head,
        row=start,
        end=end,
        H=resident_heads,
        D=DK,
        TMA=TMA_QK,
        NW=NW,
    )
    if HAS_DS:
        _load_tile(
            ptr=resident_value,
            desc=resident_value_desc,
            smem=resident_b,
            bar=bars.index(2),
            head=resident_head,
            row=start,
            end=end,
            H=resident_heads,
            D=DV,
            TMA=TMA_V,
            NW=NW,
        )
    if TMA_QK or (HAS_DS and TMA_V):
        mbarrier.wait(bars.index(2), 0)
    if not TMA_QK or (HAS_DS and not TMA_V):
        fence_async_shared()

    sl: gl.constexpr = _acc_layout(M=BM, N=BN, TCGEN=TCGEN, NW=NW)
    rows = start + gl.arange(0, BM, gl.SliceLayout(1, sl)).to(gl.int64)
    cols_local = gl.arange(0, BN, gl.SliceLayout(0, sl)).to(gl.int64)
    REUSE_GRAD: gl.constexpr = TCGEN and GRAD == 'dkv' and BK + BV > 512
    if HAS_DS:
        grad_acc = _acc_alloc(M=BM, N=BK, TCGEN=TCGEN, NW=NW)
    if HAS_DV:
        dv_acc = _acc_alloc(M=BM, N=BV, TCGEN=TCGEN, NW=NW)
    if REUSE_GRAD:
        # preserve a small gradient slice while its TMEM holds the score and probability gradient.
        scratch = grad_acc if BK >= BV else dv_acc
        score_acc = scratch.slice(0, BN)
        dp_acc = scratch.slice(BN, BN)
    else:
        score_acc = _acc_alloc(M=BM, N=BN, TCGEN=TCGEN, NW=NW)
        if HAS_DS:
            dp_acc = _acc_alloc(M=BM, N=BN, TCGEN=TCGEN, NW=NW)
    if USE_GATE:
        gate_row = gl.load(GATE + rows * HQ + hq, rows < end, other=0)
        dg = gl.full([BM], 0, gl.float32, gl.SliceLayout(1, sl))
    if USE_SINK and IS_DQ:
        sink_expectation = gl.full([BM], 0., gl.float32, gl.SliceLayout(1, sl))
    if IS_DQ:
        lse = gl.load(LSE + rows * HQ + hq, rows < end, other=0)
        delta = gl.load(DELTA + rows * HQ + hq, rows < end, other=0)
        first = bos
        if W is not None:
            first = bos + gl.maximum((start - bos - W + 1) // BN, 0) * BN
        last = gl.minimum(start + BM, end)
    else:
        first = start
        last = end
        if W is not None:
            last = gl.minimum(last, start + BM + W - 1)
    phase = 0
    for step in range(gl.cdiv(gl.maximum(last - first, 0), BN)):
        slot = (step % BUFFERS).to(gl.int32)
        stream_start = first + step * BN
        if step == 0 or BUFFERS == 1:
            _load_pair(
                A=stream_qk,
                AD=stream_qk_desc,
                AS=stream_a.index(slot),
                B=stream_value,
                BD=stream_value_desc,
                BS=stream_b.index(slot),
                bar=bars.index(slot),
                head=stream_head,
                row=stream_start,
                end=end,
                H=stream_heads,
                DA=DK,
                DB=DV,
                TMA_A=TMA_QK,
                TMA_B=TMA_V,
                NW=NW,
            )
        if TMA_QK or TMA_V:
            mbarrier.wait(bars.index(slot), ((step // BUFFERS) % 2).to(gl.int32))
        if not TMA_QK or not TMA_V:
            fence_async_shared()
        if BUFFERS > 1 and stream_start + BN < last:
            nxt = ((step + 1) % BUFFERS).to(gl.int32)
            _load_pair(
                A=stream_qk,
                AD=stream_qk_desc,
                AS=stream_a.index(nxt),
                B=stream_value,
                BD=stream_value_desc,
                BS=stream_b.index(nxt),
                bar=bars.index(nxt),
                head=stream_head,
                row=stream_start + BN,
                end=end,
                H=stream_heads,
                DA=DK,
                DB=DV,
                TMA_A=TMA_QK,
                TMA_B=TMA_V,
                NW=NW,
            )
        if REUSE_GRAD:
            saved_grad = scratch.slice(0, 2 * BN).load(_acc_layout(M=BM, N=2 * BN, TCGEN=TCGEN, NW=NW))
            # the MMA warp must not overwrite a slice while another warp is still saving it.
            barrier()
        if HAS_DS:
            score_acc, dp_acc, phase = _mma_pair(
                a=resident_a.reshape([BM, BK]),
                b=stream_a.index(slot).reshape([BN, BK]).permute((1, 0)),
                c=resident_b.reshape([BM, BV]),
                d=stream_b.index(slot).reshape([BN, BV]).permute((1, 0)),
                acc_a=score_acc,
                acc_b=dp_acc,
                bar=bars.index(3),
                phase=phase,
                TCGEN=TCGEN,
            )
        else:
            score_acc, phase = _mma(
                a=resident_a.reshape([BM, BK]),
                b=stream_a.index(slot).reshape([BN, BK]).permute((1, 0)),
                acc=score_acc,
                bar=bars.index(3),
                phase=phase,
                TCGEN=TCGEN,
                USE_ACC=False,
            )
        scores = _acc_read(acc=score_acc, M=BM, N=BN, TCGEN=TCGEN, NW=NW) * (SCALE * 1.4426950216)
        cols = stream_start + cols_local
        if IS_DQ:
            needs_mask = (stream_start + BN > start) | (start + BM > end)
            norm = lse[:, None]
            delta_b = delta[:, None]
        else:
            needs_mask = (start + BM > stream_start) | (stream_start + BN > end)
            lse_col = gl.load(LSE + cols * HQ + hq, cols < end, other=0)
            delta_col = gl.load(DELTA + cols * HQ + hq, cols < end, other=0)
            norm = lse_col[None, :]
            delta_b = delta_col[None, :]
        if USE_GATE:
            gate_col = gl.load(GATE + cols * HQ + hq, cols < end, other=0)
            if IS_DQ:
                scores += gate_row[:, None] - gate_col[None, :]
            else:
                scores += gate_col[None, :] - gate_row[:, None]
        prob = gl.exp2(scores - norm)
        if W is not None or needs_mask:
            if IS_DQ:
                distance = rows[:, None] - cols[None, :]
            else:
                distance = cols[None, :] - rows[:, None]
            mask = (distance >= 0) & (cols[None, :] < end) & (rows[:, None] < end)
            if W is not None:
                mask &= distance < W
            prob = gl.where(mask, prob, 0.)
        if HAS_DS:
            dp = _acc_read(acc=dp_acc, M=BM, N=BN, TCGEN=TCGEN, NW=NW)
            if USE_SINK and IS_DQ:
                sink_expectation += gl.sum(prob * dp, 1)
            ds = prob * (dp - delta_b)
            if not USE_SINK:
                # a single-key softmax has zero score gradient, independent of reduction roundoff.
                single_key = rows[:, None] == bos if IS_DQ else cols[None, :] == bos
                if W == 1:
                    single_key = gl.full([BM, BN], True, gl.int1, sl)
                ds = gl.where(single_key, 0., ds)
            if USE_GATE:
                dg += gl.sum(ds, 1) * (1 if IS_DQ else -1)
            ds_shared.store(ds.to(dtype))
        if HAS_DV:
            p_shared.store(prob.to(dtype))
        fence_async_shared()
        if REUSE_GRAD:
            scratch.slice(0, 2 * BN).store(saved_grad)
            # all warps must finish restoring the accumulator before the MMA warp reads it.
            barrier()
        if GRAD == 'dkv':
            grad_acc, dv_acc, phase = _mma_pair(
                a=ds_shared,
                b=stream_a.index(slot).reshape([BN, BK]),
                c=p_shared,
                d=stream_b.index(slot).reshape([BN, BV]),
                acc_a=grad_acc,
                acc_b=dv_acc,
                bar=bars.index(3),
                phase=phase,
                TCGEN=TCGEN,
                USE_ACC=True,
            )
        elif HAS_DS:
            grad_acc, phase = _mma(
                a=ds_shared,
                b=stream_a.index(slot).reshape([BN, BK]),
                acc=grad_acc,
                bar=bars.index(3),
                phase=phase,
                TCGEN=TCGEN,
            )
        else:
            dv_acc, phase = _mma(
                a=p_shared,
                b=stream_b.index(slot).reshape([BN, BV]),
                acc=dv_acc,
                bar=bars.index(3),
                phase=phase,
                TCGEN=TCGEN,
            )

    if USE_SINK and IS_DQ:
        sink_grad = -gl.exp2(gl.load(SINK + hq) - lse) * sink_expectation
        gl.store(DSINK + rows * HQ + hq, sink_grad, rows < end)
    if HAS_DS:
        grad = _acc_read(acc=grad_acc, M=BM, N=BK, TCGEN=TCGEN, NW=NW) * SCALE
        grad_layout: gl.constexpr = _acc_layout(M=BM, N=BK, TCGEN=TCGEN, NW=NW)
        gr = start + gl.arange(0, BM, gl.SliceLayout(1, grad_layout)).to(gl.int64)
        gc = gl.arange(0, BK, gl.SliceLayout(0, grad_layout)).to(gl.int64)
        target = DQ if IS_DQ else DK_OUT
        gl.store(target + (gr[:, None] * HQ + hq) * DK + gc[None, :], grad, (gr[:, None] < end) & (gc[None, :] < DK))
        if USE_GATE:
            gl.store(DG + rows * HQ + hq, dg, rows < end)
    if HAS_DV:
        grad_v = _acc_read(acc=dv_acc, M=BM, N=BV, TCGEN=TCGEN, NW=NW)
        v_layout: gl.constexpr = _acc_layout(M=BM, N=BV, TCGEN=TCGEN, NW=NW)
        vr = start + gl.arange(0, BM, gl.SliceLayout(1, v_layout)).to(gl.int64)
        vc = gl.arange(0, BV, gl.SliceLayout(0, v_layout)).to(gl.int64)
        gl.store(DV_OUT + (vr[:, None] * HQ + hq) * DV + vc[None, :], grad_v, (vr[:, None] < end) & (vc[None, :] < DV))
    for i in gl.static_range(4):
        mbarrier.invalidate(bars.index(i))


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

    b, t, hq, dk = q.shape
    h, dv = k.shape[2], v.shape[-1]
    bk, bv = max(64, triton.next_power_of_2(dk)), max(32, triton.next_power_of_2(dv))
    tcgen = get_device_capability(q.device.index)[0] == 10
    nw = 4 if tcgen else 8
    if not tcgen and 64 < max(dk, dv) <= 128:
        nw = 4
    bm, bn, buffers = (64, 32, 1) if max(dk, dv) > 256 else (64, 64, 2)
    chunk = 128 if chunk_indices is not None else bm
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=chunk)
    nt = triton.cdiv(t, bm) if cu_seqlens is None else len(chunk_indices) * (chunk // bm)
    delta = parallel_attn_bwd_preprocess(o=o, do=do)
    temp_dtype = q.dtype if h == hq else torch.float32
    dq = torch.empty_like(q)
    dk_out = torch.empty(b, t, hq, dk, device=q.device, dtype=temp_dtype)
    dv_out = torch.empty(b, t, hq, dv, device=q.device, dtype=temp_dtype)
    dg_q = torch.empty_like(lse) if g_cumsum is not None else None
    dg_k = torch.empty_like(lse) if g_cumsum is not None else None
    dsink_rows = torch.empty_like(lse) if sink_bias is not None else None
    tma_qk = IS_TMA_SUPPORTED and dk % 8 == 0 and q.data_ptr() % 16 == 0 and k.data_ptr() % 16 == 0
    tma_v = IS_TMA_SUPPORTED and dv % 8 == 0 and v.data_ptr() % 16 == 0
    tma_v = tma_v and do.data_ptr() % 16 == 0
    gradients = ('dq', 'dkv') if tcgen or bk + bv <= 512 else ('dq', 'dk', 'dv')
    q_desc, k_desc = (_descriptor(x=q, rows=bm, dim=bk), _descriptor(x=k, rows=bn, dim=bk)) if tma_qk else (q, k)
    v_desc, do_desc = (_descriptor(x=v, rows=bn, dim=bv), _descriptor(x=do, rows=bm, dim=bv)) if tma_v else (v, do)
    dq_descriptors = q_desc, k_desc, v_desc, do_desc
    if bm != bn:
        q_desc, k_desc = (_descriptor(x=q, rows=bn, dim=bk), _descriptor(x=k, rows=bm, dim=bk)) if tma_qk else (q, k)
        v_desc, do_desc = (_descriptor(x=v, rows=bm, dim=bv), _descriptor(x=do, rows=bn, dim=bv)) if tma_v else (v, do)
    dkv_descriptors = q_desc, k_desc, v_desc, do_desc
    for grad in gradients:
        q_desc, k_desc, v_desc, do_desc = dq_descriptors if grad == 'dq' else dkv_descriptors
        parallel_attn_bwd_kernel_gluon[(nt * b * hq,)](
            Q=q,
            K=k,
            V=v,
            DO=do,
            LSE=lse,
            DELTA=delta,
            DQ=dq,
            DK_OUT=dk_out,
            DV_OUT=dv_out,
            GATE=g_cumsum,
            DG=dg_q if grad == 'dq' else dg_k,
            SINK=sink_bias,
            DSINK=dsink_rows,
            CU=cu_seqlens,
            INDICES=chunk_indices,
            Q_DESC=q_desc,
            K_DESC=k_desc,
            V_DESC=v_desc,
            DO_DESC=do_desc,
            T=t,
            NT=nt,
            H=h,
            HQ=hq,
            DK=dk,
            DV=dv,
            SCALE=scale,
            W=window_size,
            BM=bm,
            BN=bn,
            BK=bk,
            BV=bv,
            CHUNK=chunk,
            BUFFERS=buffers,
            TCGEN=tcgen,
            TMA_QK=tma_qk,
            TMA_V=tma_v,
            VARLEN=cu_seqlens is not None,
            USE_GATE=g_cumsum is not None,
            USE_SINK=sink_bias is not None,
            GRAD=grad,
            NW=nw,
            num_warps=nw,
        )
    if hq != h:
        dk_out = reduce(dk_out, 'b t (h g) k -> b t h k', g=hq // h, reduction='sum')
        dv_out = reduce(dv_out, 'b t (h g) v -> b t h v', g=hq // h, reduction='sum')
    if g_cumsum is not None:
        dg_q.add_(dg_k)
    dsink = None if dsink_rows is None else dsink_rows.sum((0, 1))
    return dq, dk_out, dv_out, dg_q, dsink
