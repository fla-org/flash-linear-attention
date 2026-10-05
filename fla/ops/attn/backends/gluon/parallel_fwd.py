# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import inspect

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell as bw
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma

from fla.ops.attn.backends.gluon.parallel import _acc_layout, _packed_operand, _tmem_layout

WARP_SPECIALIZE_V2 = 'functions_and_args' in inspect.signature(gl.warp_specialize).parameters


@gluon.jit
def _load(args, descs, cfg: gl.constexpr):
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
def _qk(qs, ks, scores, kr, kf, sr, sf, i, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    slot = i % 2
    phase = (i // 2) % 2
    mbarrier.wait(kr.index(slot), phase)
    mbarrier.wait(sf.index(slot), phase)
    bw.tcgen05_mma(
        a=qs.reshape([BM, BK]),
        b=ks.index(slot).reshape([BN, BK]).permute((1, 0)),
        acc=scores.index(slot),
        use_acc=False,
    )
    bw.tcgen05_commit(sr.index(slot))
    bw.tcgen05_commit(kf.index(slot))


@gluon.jit
def _mma(args, descs, cfg: gl.constexpr):
    qs, ks, vs, scores, output, bars, positions, pointers = args
    qb, kr, kf, vr, vf, sr, sf, pr, ore, ordy = bars
    hq, hk, start, end, first, steps, iv = positions
    BM: gl.constexpr = cfg[0]
    BN: gl.constexpr = cfg[1]
    BK: gl.constexpr = cfg[2]
    BV: gl.constexpr = cfg[3]
    mbarrier.wait(qb, 0)
    if steps > 0:
        _qk(qs=qs, ks=ks, scores=scores, kr=kr, kf=kf, sr=sr, sf=sf, i=0, BM=BM, BN=BN, BK=BK)
    for i in range(steps):
        # the next QK runs while the compute partition normalizes the current scores.
        if i + 1 < steps:
            _qk(qs=qs, ks=ks, scores=scores, kr=kr, kf=kf, sr=sr, sf=sf, i=i + 1, BM=BM, BN=BN, BK=BK)
        slot = i % 2
        phase = (i // 2) % 2
        mbarrier.wait(pr.index(slot), phase)
        mbarrier.wait(ore, i % 2)
        mbarrier.wait(vr.index(slot), phase)
        prob = _packed_operand(score=scores.index(slot), dtype=qs.dtype, BM=BM, BN=BN)
        bw.tcgen05_mma(a=prob, b=vs.index(slot).reshape([BN, BV]), acc=output, use_acc=i > 0)
        bw.tcgen05_commit(ordy)
        bw.tcgen05_commit(sf.index(slot))
        bw.tcgen05_commit(vf.index(slot))


@gluon.jit
def _compute(args, descs, cfg: gl.constexpr):
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
        mask = (rows[:, None] >= cols[None, :]) & (cols[None, :] < end)
        if W is not None:
            mask &= rows[:, None] - cols[None, :] < W
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
    Q_DESC, K_DESC, V_DESC, O, LSE, GATE, SINK, CU, INDICES,
    T, H: gl.constexpr, HQ: gl.constexpr, DV: gl.constexpr,
    SCALE: gl.constexpr, W: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
    BK: gl.constexpr, BV: gl.constexpr, VARLEN: gl.constexpr,
    USE_GATE: gl.constexpr, USE_SINK: gl.constexpr, WS_V2: gl.constexpr,
):
    iv = gl.program_id(0).to(gl.int64)
    it = gl.program_id(1).to(gl.int64)
    bh = gl.program_id(2).to(gl.int64)
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
    first = bos
    if W is not None:
        first = bos + gl.maximum((start - bos - W + 1) // BN, 0) * BN
    steps = gl.cdiv(gl.maximum(gl.minimum(start + BM, end) - first, 0), BN).to(gl.int32)
    dtype: gl.constexpr = O.dtype.element_ty
    qs = gl.allocate_shared_memory(dtype, [1, BM, BK], Q_DESC.layout)
    ks = gl.allocate_shared_memory(dtype, [2, 1, BN, BK], K_DESC.layout)
    vs = gl.allocate_shared_memory(dtype, [2, 1, BN, BV], V_DESC.layout)
    scores = bw.allocate_tensor_memory(gl.float32, [2, BM, BN], _tmem_layout(M=BM, N=BN))
    output = bw.allocate_tensor_memory(gl.float32, [BM, BV], _tmem_layout(M=BM, N=BV))
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
            functions_and_args=[(_compute, (args, descs, cfg)), (_mma, (args, descs, cfg)), (_load, (args, descs, cfg))],
            worker_num_warps=[1, 1],
            worker_num_regs=[24, 24],
        )
    else:
        gl.warp_specialize(
            default_args=(args, descs, cfg),
            default_partition=_compute,
            worker_args=(args, descs, cfg),
            worker_partitions=[_mma, _load],
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
