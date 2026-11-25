# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

"""Chunk preparation for the chunked Gated DeltaNet forward (CuTe DSL, tcgen05 Gram).

Per 64-token chunk of one sequence and one value head h (gamma = in-chunk
inclusive cumsum of g, invalid tail tokens get g = beta = 0, k = q = 0):

    L[t, s]   = beta_t exp(gamma_t - gamma_s) k_t . k_s          (s < t)
    A1        = (I + L)^-1 diag(beta)                             (bf16)
    Mqk[t, s] = scale exp(gamma_t - gamma_s) q_t . k_s            (s <= t, bf16)
    fac       = [exp(gamma_t) (64), exp(gamma_C - gamma_t) (64), exp(gamma_C), 0...]

Chunk slots: chunk c of sequence s lives in slot cu[s] // 64 + s + c (dense,
collision free, < T // 64 + N).  One CTA per (slot, q/k head) computes the
Gram matrices once and loops over the Hv / Hq value heads of its group.

A1 and Mqk are lower triangular: with `cmp` (tcgen05 consumers) only their 10 lower
16x16 blocks are stored (compact, 2560 elements per matrix: row block rb keeps columns
[0, 16 (rb + 1)) row-major at offset 128 rb (rb + 1)); the tcgen05 kernel's TMA
zero-fills the upper blocks.  The mma.sync recurrence reads full 64x64 tiles.

The Gram [Gk; Gq] = [K; Q] K^T is one tcgen05 UMMA chain (M 128, N 64, K 128)
from a K-major SW128 [K; Q] staging tile into 64 TMEM columns; thread t reads
TMEM lane t (t < 64: Gk row t -> L row t; t >= 64: Gq row t - 64 -> Mqk row).
The inverse uses warp-level TF32 mma.sync merges.
"""

import cutlass
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05, warp
from cutlass.experimental.primitives.nvvm_wrapper import tcgen05_fence

C = 64
D = 128
LOG2E = 1.4426950408889634
PITCH = 68
TPITCH = 40


CTRI = 2560


def tri_index(rb, r, cc):
    """Compact element index of (row r, column cc) in row block rb (a Python int)."""
    return 128 * rb * (rb + 1) + (r - 16 * rb) * (16 * (rb + 1)) + cc


def kmajor(rows, cols):
    atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16)
    return cute.tile_to_shape(atom, (rows, cols), (0, 1))


@cute.struct
class Bars:
    mb: cute.struct.MemRange[cutlass.Int64, 1]
    tmem_slot: cutlass.Int32


@cute.jit
def tf32_product(
    tiled: cute.TiledMma,
    thr_idx,
    sA: cute.Tensor,
    sB: cute.Tensor,
    sC: cute.Tensor,
    NEG: cutlass.Constexpr,
):
    """sC = (+/-) sA @ sB^T-view: sA (M, K), sB (N, K), sC (M, N), fp32 smem."""
    thr = tiled.get_slice(thr_idx)
    sA32 = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=cutlass.Int32), sA.layout)
    sB32 = cute.make_tensor(cute.recast_ptr(sB.iterator, dtype=cutlass.Int32), sB.layout)
    tA = thr.partition_A(sA32)
    tB = thr.partition_B(sB32)
    tC = thr.partition_C(sC)
    rA = thr.make_fragment_A(tA)
    rB = thr.make_fragment_B(tB)
    rA.store(tA.load())
    rB.store(tB.load())
    acc = cute.make_rmem_tensor(thr.partition_shape_C(cute.shape(sC)), cutlass.Float32)
    acc.fill(0.0)
    for k in cutlass.range_constexpr(cute.size(rA.shape[2])):
        cute.gemm(tiled, acc, rA[None, None, k], rB[None, None, k], acc)
    if cutlass.const_expr(NEG):
        tC.store(-acc.load())
    else:
        tC.store(acc.load())


@cute.jit
def gate_l4(rG: cute.Tensor, j: cutlass.Constexpr, a, sEi: cute.Tensor, sL, rowt, dg):
    """L row rowt, columns [4j, 4j + 4): Gk * (e_t beta_t) * ei_s, zero for s >= t;
    returns the diagonal value if it lies in this group (else dg)."""
    ei4 = cute.make_rmem_tensor((4,), cutlass.Float32)
    lv = cute.make_rmem_tensor((4,), cutlass.Float32)
    cute.autovec_copy(cute.make_tensor(sEi.iterator + 4 * j, cute.make_layout(4)), ei4)
    for e in cutlass.range_constexpr(4):
        v = rG[4 * j + e] * a * ei4[e]
        if 4 * j + e == rowt:
            dg = v
        if 4 * j + e >= rowt:
            v = cutlass.Float32(0.0)
        lv[e] = v
    cute.make_tensor(sL.iterator + cute.assume(rowt * PITCH + 4 * j, divby=4), cute.make_layout(4)).store(lv.load())
    return dg


@cute.jit
def gate_l4_exact(rG: cute.Tensor, j: cutlass.Constexpr, gt, bt, sGam, cb, sL, rowt, dg):
    """L row rowt, columns [4j, 4j + 4) with per-element exp2(gamma_t - gamma_s) (|gamma| > 100)."""
    lv = cute.make_rmem_tensor((4,), cutlass.Float32)
    for e in cutlass.range_constexpr(4):
        v = cutlass.Float32(0.0)
        if 4 * j + e < rowt:
            dec = cute.math.exp2(gt - sGam[cb + 4 * j + e], fastmath=True)
            v = rG[4 * j + e] * dec * bt
        if 4 * j + e == rowt:
            dg = rG[4 * j + e] * bt
        lv[e] = v
    cute.make_tensor(sL.iterator + cute.assume(rowt * PITCH + 4 * j, divby=4), cute.make_layout(4)).store(lv.load())
    return dg


@cute.jit
def gate_m8(rG: cute.Tensor, j: cutlass.Constexpr, a, sEi: cute.Tensor, sMb, rowt, mrow):
    """Mqk row rowt, columns [8j, 8j + 8): Gq * (scale e_t) * ei_s, zero for s > t -> bf16."""
    ei8 = cute.make_rmem_tensor((8,), cutlass.Float32)
    mv = cute.make_rmem_tensor((8,), cutlass.Float32)
    cute.autovec_copy(cute.make_tensor(sEi.iterator + 8 * j, cute.make_layout(8)), ei8)
    for e in cutlass.range_constexpr(8):
        v = rG[8 * j + e] * a * ei8[e]
        if 8 * j + e > rowt:
            v = cutlass.Float32(0.0)
        mv[e] = v
    m16 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    m16.store(mv.load().to(cutlass.BFloat16))
    cute.autovec_copy(
        m16,
        cute.make_tensor(
            sMb.iterator + cute.assume(mrow + 8 * j, divby=8),
            cute.make_layout(8),
        ),
    )


@cute.jit
def zero_m8(j: cutlass.Constexpr, sMb, mrow):
    m16 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    m16.fill(0.0)
    cute.autovec_copy(
        m16,
        cute.make_tensor(
            sMb.iterator + cute.assume(mrow + 8 * j, divby=8),
            cute.make_layout(8),
        ),
    )


@cute.jit
def gates_warp(lane, g0, g1, sGam, sE, sEi, sAbs, FAC, fac_base, cb, ab):
    """One warp, one value head: gamma = inclusive cumsum of g (log2 units) for tokens
    2 lane, 2 lane + 1, the exp tables (buffer offset cb), max |gamma| (sAbs[ab]) and the
    head's factors exp(gamma_t), exp(gamma_C - gamma_t), exp(gamma_C)."""
    t0 = lane * 2
    pair = g0 + g1
    incl = pair
    for off in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_up(incl, 1 << off)
        if lane >= (1 << off):
            incl = incl + other
    excl = incl - pair
    ga = (excl + g0) * LOG2E
    gb = incl * LOG2E
    sGam[cb + t0] = ga
    sGam[cb + t0 + 1] = gb
    gabs = cute.arch.fmax(cute.math.abs(ga), cute.math.abs(gb))
    gabs = cute.arch.warp_reduction_max(gabs)
    gcw = cute.arch.shuffle_sync(gb, 31)
    ea = cute.math.exp2(ga, fastmath=True)
    eb = cute.math.exp2(gb, fastmath=True)
    sE[cb + t0] = ea
    sE[cb + t0 + 1] = eb
    sEi[cb + t0] = cute.math.exp2(-ga, fastmath=True)
    sEi[cb + t0 + 1] = cute.math.exp2(-gb, fastmath=True)
    FAC[fac_base + t0] = ea
    FAC[fac_base + t0 + 1] = eb
    FAC[fac_base + C + t0] = cute.math.exp2(gcw - ga, fastmath=True)
    FAC[fac_base + C + t0 + 1] = cute.math.exp2(gcw - gb, fastmath=True)
    if lane == 0:
        sAbs[ab] = gabs
        FAC[fac_base + 2 * C] = cute.math.exp2(gcw, fastmath=True)


@cute.jit
def fac_tail(lane, FAC, fac_base):
    """One warp: unused factor tail [2C + 3, 3C) (2C + 1, 2C + 2 are written later)."""
    FAC[fac_base + 2 * C + 3 + lane] = cutlass.Float32(0.0)
    if lane < C - 35:
        FAC[fac_base + 2 * C + 35 + lane] = cutlass.Float32(0.0)


@cute.kernel
def prep_kernel(
    K: cute.Tensor,
    Q: cute.Tensor,
    G: cute.Tensor,
    Beta: cute.Tensor,
    Cu: cute.Tensor,
    A1: cute.Tensor,
    MQK: cute.Tensor,
    FAC: cute.Tensor,
    nseq: cutlass.Int32,
    HQ: cutlass.Int32,
    HV: cutlass.Int32,
    dense_t: cutlass.Int32,
    scale: cutlass.Float32,
    gram: cute.TiledMma,
    mma16: cute.TiledMma,
    mma32: cute.TiledMma,
    gmem_copy: cute.TiledCopy,
    kq_layout: cute.ComposedLayout,
    GQA: cutlass.Constexpr,
    CMP: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    slot, hq, _ = cute.arch.block_idx()
    lane = tid % 32
    warp_u = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    grp = HV // HQ

    cnt = cutlass.Int32(0)
    for it in range((nseq + 31) // 32):
        i = it * 32 + lane
        hit = cutlass.Int32(0)
        if i < nseq:
            b_i = i * dense_t
            if dense_t == 0:
                b_i = Cu[i]
            if b_i // C + i <= slot:
                hit = 1
        cnt = cnt + cute.arch.popc(cute.arch.vote_ballot_sync(hit == 1))
    seq = cnt - 1
    bos = seq * dense_t
    eos = bos + dense_t
    if dense_t == 0:
        bos = Cu[seq]
        eos = Cu[seq + 1]
    chunk = slot - (bos // C + seq)
    start = bos + chunk * C
    if start < eos:
        length = eos - start
        length = min(length, C)

        smem = cutlass.utils.SmemAllocator()

        NG = 2 if GQA else 1

        sLXT = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(2 * C * PITCH + 32 * TPITCH),
            byte_alignment=1024,
        )
        bars = smem.allocate(Bars)
        sKQ = cute.make_tensor(
            cute.recast_ptr(sLXT.iterator, swizzle_=kq_layout.inner, dtype=cutlass.BFloat16),
            kq_layout.outer,
        )
        sK = cute.local_tile(sKQ, (C, D), (0, 0))
        sQ = cute.local_tile(sKQ, (C, D), (1, 0))
        lay = cute.make_layout((C, PITCH), stride=(PITCH, 1))
        sL = cute.make_tensor(sLXT.iterator, lay)
        sX = cute.make_tensor(sLXT.iterator + C * PITCH, lay)

        sT = cute.make_tensor(
            sLXT.iterator + 2 * C * PITCH,
            cute.make_layout((32, TPITCH), stride=(TPITCH, 1)),
        )

        sMb = cute.make_tensor(cute.recast_ptr(sLXT.iterator, dtype=cutlass.BFloat16), cute.make_layout(1))
        sGam = smem.allocate_tensor(cutlass.Float32, cute.make_layout(NG * C), 16)
        L16 = cute.make_layout((16, 16), stride=(PITCH, 1))
        L16T = cute.make_layout((16, 16), stride=(1, PITCH))
        L32 = cute.make_layout((32, 32), stride=(PITCH, 1))
        L32T = cute.make_layout((32, 32), stride=(1, PITCH))
        T16 = cute.make_layout((16, 16), stride=(TPITCH, 1))
        T16T = cute.make_layout((16, 16), stride=(1, TPITCH))
        T32 = cute.make_layout((32, 32), stride=(TPITCH, 1))
        T32T = cute.make_layout((32, 32), stride=(1, TPITCH))
        st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
        sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout(NG * C), 16)
        sE = smem.allocate_tensor(cutlass.Float32, cute.make_layout(NG * C), 16)
        sEi = smem.allocate_tensor(cutlass.Float32, cute.make_layout(NG * C), 16)

        sAbs = smem.allocate_tensor(cutlass.Float32, cute.make_layout(4), 16)
        sRed = smem.allocate_tensor(cutlass.Float32, cute.make_layout(4), 16)
        mb = bars.mb.data_ptr()
        if tid == 0:
            cute.arch.mbarrier_init(mb, 1)
        cute.arch.mbarrier_init_fence()

        row_stride = HQ * D
        gK = cute.make_tensor(
            K.iterator + cute.assume(start * row_stride + hq * D, divby=8),
            cute.make_layout((C, D), stride=(cute.assume(row_stride, divby=128), 1)),
        )
        gQ = cute.make_tensor(
            Q.iterator + cute.assume(start * row_stride + hq * D, divby=8),
            cute.make_layout((C, D), stride=(cute.assume(row_stride, divby=128), 1)),
        )
        thr_copy = gmem_copy.get_slice(tid)
        tKg = thr_copy.partition_S(gK)
        tKs = thr_copy.partition_D(sK)
        tQg = thr_copy.partition_S(gQ)
        tQs = thr_copy.partition_D(sQ)
        cKV = cute.make_identity_tensor((C, D))
        tc = thr_copy.partition_S(cKV)
        for m in cutlass.range_constexpr(cute.size(tKs.shape[1])):
            if tc[0, m, 0][0] < length:
                cute.copy(gmem_copy, tKg[None, m, None], tKs[None, m, None])
                cute.copy(gmem_copy, tQg[None, m, None], tQs[None, m, None])
            else:
                tKs[None, m, None].fill(0)
                tQs[None, m, None].fill(0)
        cute.arch.cp_async_commit_group()

        pg0 = cutlass.Float32(0.0)
        pg1 = cutlass.Float32(0.0)
        pb = cutlass.Float32(0.0)
        if tid < 32:
            if lane * 2 < length:
                pg0 = G[(start + lane * 2) * HV + hq * grp]
            if lane * 2 + 1 < length:
                pg1 = G[(start + lane * 2 + 1) * HV + hq * grp]
        elif tid < 96:
            if tid - 32 < length:
                pb = Beta[(start + tid - 32) * HV + hq * grp]

        tmem = utils.TmemAllocator(
            bars.tmem_slot.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128),
        )
        tmem.allocate(64)

        tmem.relinquish_alloc_permit()

        if tid < 32:
            gates_warp(
                lane,
                pg0,
                pg1,
                sGam,
                sE,
                sEi,
                sAbs,
                FAC,
                slot * HV * 192 + hq * grp * 192,
                0,
                0,
            )
        elif tid < 96:
            sBeta[tid - 32] = pb
        else:
            fac_tail(lane, FAC, (slot * HV + hq * grp) * 192)
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        tcgen05_fence("before_thread_sync")
        cute.arch.sync_threads()
        tcgen05_fence("after_thread_sync")
        tptr = tmem.retrieve_ptr(cutlass.Float32)
        thr_g = gram.get_slice(0)
        acc_layout = gram.make_fragment_C(gram.partition_shape_C((2 * C, C))).layout
        tG = cute.make_tensor(tptr, acc_layout)
        ld = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32),
            tG,
        )
        ld_t = ld.get_slice(tid)
        tG_src = ld_t.partition_S(tG)
        cG = ld_t.partition_D(thr_g.partition_C(cute.make_identity_tensor((2 * C, C))))
        rG = cute.make_rmem_tensor(cG.shape, cutlass.Float32)

        if warp_u == 0:
            fa = gram.make_fragment_A(thr_g.partition_A(sKQ))
            fb = gram.make_fragment_B(thr_g.partition_B(sK))
            tcgen05_fence("after_thread_sync")
            gram.set(tcgen05.Field.ACCUMULATE, False)
            for k in cutlass.range_constexpr(D // 16):
                cute.gemm(gram, tG, fa[None, None, k], fb[None, None, k], tG)
                gram.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mb)

        rowt = tid % C
        for hh in range(grp):
            h = hq * grp + hh
            fac_base = (slot * HV + h) * 192
            cb = 0
            if cutlass.const_expr(GQA):
                cb = (hh % 2) * C
                if hh > 0:
                    cute.arch.sync_threads()
                gam_c = sGam[cb + C - 1]
                gam_abs = sAbs[hh % 2]

                ng0 = cutlass.Float32(0.0)
                ng1 = cutlass.Float32(0.0)
                if hh + 1 < grp:
                    if warp_u == 2:
                        if lane * 2 < length:
                            ng0 = G[(start + lane * 2) * HV + h + 1]
                        if lane * 2 + 1 < length:
                            ng1 = G[(start + lane * 2 + 1) * HV + h + 1]
                    elif warp_u == 3:
                        if lane * 2 < length:
                            ng0 = Beta[(start + lane * 2) * HV + h + 1]
                        if lane * 2 + 1 < length:
                            ng1 = Beta[(start + lane * 2 + 1) * HV + h + 1]
            else:
                gam_c = sGam[C - 1]
                gam_abs = sAbs[0]

            cute.arch.mbarrier_wait(mb, 0)
            tcgen05_fence("after_thread_sync")
            if hh == 0:
                for j in cutlass.range_constexpr(4):
                    idx = j * 128 + tid
                    blk = idx // 256
                    sX[blk * 32 + (idx % 256) // 16, blk * 32 + 16 + idx % 16] = cutlass.Float32(0.0)
            cute.copy(ld, tG_src, rG)
            out_base = (slot * HV + h) * (C * C)
            if cutlass.const_expr(CMP):
                out_base = (slot * HV + h) * CTRI

            e_t = sE[cb + rowt]
            sEi_h = sEi
            if cutlass.const_expr(GQA):
                sEi_h = cute.make_tensor(sEi.iterator + cute.assume(cb, divby=64), cute.make_layout(C))
            if warp_u < 2:
                bt = sBeta[cb + rowt]
                dg = cutlass.Float32(0.0)
                if gam_abs <= 100.0:
                    a = e_t * bt

                    for j in cutlass.range_constexpr(8):
                        dg = gate_l4(rG, j, a, sEi_h, sL, rowt, dg)
                    if warp_u == 1:
                        for j in cutlass.range_constexpr(8, 16):
                            dg = gate_l4(rG, j, a, sEi_h, sL, rowt, dg)
                else:
                    gt = sGam[cb + rowt]

                    for j in cutlass.range_constexpr(8):
                        dg = gate_l4_exact(rG, j, gt, bt, sGam, cb, sL, rowt, dg)
                    if warp_u == 1:
                        for j in cutlass.range_constexpr(8, 16):
                            dg = gate_l4_exact(rG, j, gt, bt, sGam, cb, sL, rowt, dg)
                dm = cute.arch.warp_reduction_max(dg)
                if lane == 0:
                    sRed[warp_u] = dm

                cute.arch.sync_warp()
                xs = cute.make_rmem_tensor((16,), cutlass.Float32)
                b0 = 32 * warp_u + 16 * (lane // 16)
                cl = lane % 16
                for i in cutlass.range_constexpr(16):
                    part = cute.make_rmem_tensor((4,), cutlass.Float32)
                    part.fill(0.0)
                    for gq in cutlass.range_constexpr((i + 3) // 4):
                        lrow = cute.make_rmem_tensor((4,), cutlass.Float32)
                        cute.autovec_copy(
                            cute.make_tensor(
                                sL.iterator + cute.assume((b0 + i) * PITCH + b0 + 4 * gq, divby=4),
                                cute.make_layout(4),
                            ),
                            lrow,
                        )
                        for e in cutlass.range_constexpr(4):
                            if 4 * gq + e < i:
                                part[e] = part[e] - lrow[e] * xs[4 * gq + e]
                    acc = (part[0] + part[1]) + (part[2] + part[3])
                    if cl == i:
                        acc = acc + cutlass.Float32(1.0)
                    xs[i] = acc
                for i in cutlass.range_constexpr(16):
                    sX[b0 + i, b0 + cl] = xs[i]
                cute.arch.sync_warp()

                o = 32 * warp_u
                tf32_product(
                    mma16,
                    lane,
                    cute.make_tensor(sL.iterator + (o + 16) * PITCH + o, L16),
                    cute.make_tensor(sX.iterator + o * PITCH + o, L16T),
                    cute.make_tensor(sT.iterator + 16 * TPITCH + o // 2, T16),
                    False,
                )
                cute.arch.sync_warp()
                tf32_product(
                    mma16,
                    lane,
                    cute.make_tensor(sX.iterator + (o + 16) * PITCH + o + 16, L16),
                    cute.make_tensor(sT.iterator + 16 * TPITCH + o // 2, T16T),
                    cute.make_tensor(sX.iterator + (o + 16) * PITCH + o, L16),
                    True,
                )
            else:
                mrow = 2 * ((rowt + 32 * (rowt // 32)) * PITCH + 32)
                if gam_abs <= 100.0:
                    a = e_t * scale
                    for j in cutlass.range_constexpr(4):
                        gate_m8(rG, j, a, sEi_h, sMb, rowt, mrow)

                    if warp_u == 3:
                        for j in cutlass.range_constexpr(4, 8):
                            gate_m8(rG, j, a, sEi_h, sMb, rowt, mrow)
                    else:
                        for j in cutlass.range_constexpr(4, 8):
                            zero_m8(j, sMb, mrow)
                else:
                    gt = sGam[cb + rowt]
                    mv = cute.make_rmem_tensor((8,), cutlass.Float32)
                    m16 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                    for j in cutlass.range_constexpr(8):
                        for e in cutlass.range_constexpr(8):
                            v = cutlass.Float32(0.0)
                            if 8 * j + e <= rowt:
                                dec = cute.math.exp2(gt - sGam[cb + 8 * j + e], fastmath=True)
                                v = rG[8 * j + e] * dec * scale
                            mv[e] = v
                        m16.store(mv.load().to(cutlass.BFloat16))
                        cute.autovec_copy(
                            m16,
                            cute.make_tensor(
                                sMb.iterator + cute.assume(mrow + 8 * j, divby=8),
                                cute.make_layout(8),
                            ),
                        )

                cute.arch.barrier(barrier_id=2, number_of_threads=64)
                for j in cutlass.range_constexpr(C * C // (64 * 8)):
                    idx = (j * 64 + tid - 64) * 8
                    r = idx // C
                    cc = idx % C

                    wlim = 16 * (j // 2 + 1) if CMP else C
                    if cc < wlim:
                        stage = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                        stage.store(
                            cute.make_tensor(
                                sMb.iterator
                                + cute.assume(
                                    2 * ((r + 32 * (r // 32)) * PITCH + 32) + cc,
                                    divby=8,
                                ),
                                cute.make_layout(8),
                            ).load()
                        )
                        didx = idx
                        if cutlass.const_expr(CMP):
                            didx = tri_index(j // 2, r, cc)
                        dst = cute.make_tensor(
                            MQK.iterator + cute.assume(out_base + didx, divby=8),
                            cute.make_layout(8),
                        )
                        cute.copy(st_atom, stage, dst)
                if cutlass.const_expr(GQA):
                    if hh + 1 < grp:
                        nb = ((hh + 1) % 2) * C
                        if warp_u == 2:
                            gates_warp(
                                lane,
                                ng0,
                                ng1,
                                sGam,
                                sE,
                                sEi,
                                sAbs,
                                FAC,
                                fac_base + 192,
                                nb,
                                (hh + 1) % 2,
                            )
                        else:
                            sBeta[nb + lane * 2] = ng0
                            sBeta[nb + lane * 2 + 1] = ng1
                            fac_tail(lane, FAC, fac_base + 192)
            cute.arch.sync_threads()
            if tid == 0:
                FAC[fac_base + 2 * C + 1] = cute.arch.fmax(sRed[0], sRed[1])
                FAC[fac_base + 2 * C + 2] = gam_c

            tf32_product(
                mma32,
                tid,
                cute.make_tensor(sL.iterator + 32 * PITCH, L32),
                cute.make_tensor(sX.iterator, L32T),
                cute.make_tensor(sT.iterator, T32),
                False,
            )
            cute.arch.sync_threads()
            tf32_product(
                mma32,
                tid,
                cute.make_tensor(sX.iterator + 32 * PITCH + 32, L32),
                cute.make_tensor(sT.iterator, T32T),
                cute.make_tensor(sX.iterator + 32 * PITCH, L32),
                True,
            )
            cute.arch.sync_threads()

            hsw = (tid // 4) % 2
            for j in cutlass.range_constexpr(C * C // (128 * 8)):
                idx = (j * 128 + tid) * 8
                r = idx // C
                cc = idx % C

                wlim = 16 * (j + 1) if CMP else C
                if cc < wlim:
                    xa = cute.make_tensor(
                        sX.iterator + cute.assume(r * PITCH + cc + 4 * hsw, divby=4),
                        cute.make_layout(4),
                    ).load()
                    xb = cute.make_tensor(
                        sX.iterator + cute.assume(r * PITCH + cc + 4 - 4 * hsw, divby=4),
                        cute.make_layout(4),
                    ).load()
                    ba = cute.make_tensor(
                        sBeta.iterator + cute.assume(cb + cc + 4 * hsw, divby=4),
                        cute.make_layout(4),
                    ).load()
                    bb = cute.make_tensor(
                        sBeta.iterator + cute.assume(cb + cc + 4 - 4 * hsw, divby=4),
                        cute.make_layout(4),
                    ).load()
                    pa = cute.make_rmem_tensor((4,), cutlass.Float32)
                    pb4 = cute.make_rmem_tensor((4,), cutlass.Float32)
                    pa.store(xa * ba)
                    pb4.store(xb * bb)
                    pv = cute.make_rmem_tensor((8,), cutlass.Float32)
                    for e in cutlass.range_constexpr(4):
                        lo = pa[e]
                        hi = pb4[e]
                        if hsw == 1:
                            lo = pb4[e]
                            hi = pa[e]
                        if cutlass.const_expr(j < 2 and not CMP):
                            if cc >= 32:
                                lo = cutlass.Float32(0.0)
                                hi = cutlass.Float32(0.0)
                        pv[e] = lo
                        pv[4 + e] = hi
                    stage16 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                    stage16.store(pv.load().to(cutlass.BFloat16))
                    didx = idx
                    if cutlass.const_expr(CMP):
                        didx = tri_index(j, r, cc)
                    dst = cute.make_tensor(
                        A1.iterator + cute.assume(out_base + didx, divby=8),
                        cute.make_layout(8),
                    )
                    cute.copy(st_atom, stage16, dst)

        cute.arch.fence_view_async_tmem_load()
        tcgen05_fence("before_thread_sync")
        cute.arch.sync_threads()
        tmem.free(tptr, 64)


@cute.jit
def prep_launch(
    pK: cute.Pointer,
    pQ: cute.Pointer,
    pG: cute.Pointer,
    pBeta: cute.Pointer,
    pCu: cute.Pointer,
    pA1: cute.Pointer,
    pMQK: cute.Pointer,
    pFAC: cute.Pointer,
    ntok: cutlass.Int32,
    nslot: cutlass.Int32,
    nseq: cutlass.Int32,
    HQ: cutlass.Int32,
    HV: cutlass.Int32,
    dense_t: cutlass.Int32,
    scale: cutlass.Float32,
    minb: cutlass.Constexpr,
    gqa: cutlass.Constexpr,
    cmp: cutlass.Constexpr,
    stream,
):
    K = cute.make_tensor(pK, cute.make_layout(ntok * HQ * D))
    Q = cute.make_tensor(pQ, cute.make_layout(ntok * HQ * D))
    G = cute.make_tensor(pG, cute.make_layout(ntok * HV))
    Beta = cute.make_tensor(pBeta, cute.make_layout(ntok * HV))
    Cu = cute.make_tensor(pCu, cute.make_layout(nseq + 1))
    A1 = cute.make_tensor(pA1, cute.make_layout(nslot * HV * C * C))
    MQK = cute.make_tensor(pMQK, cute.make_layout(nslot * HV * C * C))
    FAC = cute.make_tensor(pFAC, cute.make_layout(nslot * HV * 192))
    gram = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (128, 64, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
    )
    atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    gmem_copy = cute.make_tiled_copy_tv(atom, cute.make_layout((16, 8), stride=(8, 1)), cute.make_layout((1, 8)))
    mma16 = cute.make_tiled_mma(warp.MmaTF32Op((16, 8, 8)), (1, 1, 1), permutation_mnk=(16, 16, 8))
    mma32 = cute.make_tiled_mma(warp.MmaTF32Op((16, 8, 8)), (2, 2, 1), permutation_mnk=(32, 32, 8))
    prep_kernel(
        K,
        Q,
        G,
        Beta,
        Cu,
        A1,
        MQK,
        FAC,
        nseq,
        HQ,
        HV,
        dense_t,
        scale,
        gram,
        mma16,
        mma32,
        gmem_copy,
        kmajor(2 * C, D),
        gqa,
        cmp,
    ).launch(
        grid=(nslot, HQ, 1),
        block=(128, 1, 1),
        min_blocks_per_mp=minb,
        stream=stream,
    )
