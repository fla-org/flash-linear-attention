# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

"""Warp-specialized tcgen05 recurrence with context parallelism (MODE 0/1/2).

320 threads.  Warps 0-3 = state chain (TMEM loads, bf16 conversions back into
TMEM; named barrier 2 within the state warp group), warps 4-7 = output epilogue and the upper
half of the bf16 state copy, warp 8 = tcgen05 MMA issue, warp 9 = TMA loads.  Every
MMA operand produced by the chain (W, fV', V', the bf16 state copy) is written
to TMEM (tcgen05.st, two bf16 per column) and read as the A operand from TMEM,
so these operands cost no shared-memory stores or MMA smem reads.  The chain
never issues MMAs: it hands rows to warp 8 by named-barrier arrive (BAR_W: W,
BAR_P: fV', BAR_R: V', BAR_S: bf16 state copy, also arrived by warps 4-7) and
waits on per-step MMA completion mbarriers (mbu: U, mbv: V', mbs: state
update).  O1 = S^T Q^T is issued after the V' MMA (off the critical path).
Loads: warp 9 issues TMA (q / k / v tiles, A1 / Mqk) and a bulk copy (factors)
of chunk l + STAGES into stage l % STAGES once the MMAs of chunk l completed
(sfree[l % STAGES], committed with obar: a per-stage barrier cannot run two phases
ahead of the loader, since its next phase needs the loads issued after the wait;
the double-buffered obar can, as chunk l + 2's loads precede the obar(l) wait, ), completing full[l % STAGES] by transaction count (STAGES <= 3); each TMA
issue costs ~40 cycles, so it stays off the MMA and epilogue warps.  A1 / Mqk come
compact from the prep (lower 16x16 blocks only): one TMA per 16-row block from a
(16 (r + 1))-column global view, whose 64-column box zero-fills the upper blocks.
Other handoffs: obar[b] (commit
after the O2 MMA of local chunk l, b = l % 2; warp 8 always waits it, since the
next V'A MMA rewrites the fV' / V' columns), epb[b] (chain thread 0 arrives
after writing the scale*e*lam table sEp[:, b]), ofree[b] (128 epilogue arrivals
after their TMEM loads; warp 8 waits before O1 into buffer b for chunk l + 2,
the chain before rewriting sEp[:, b] and before its state copy into the O2 half
of buffer b), sdone (commit after the state update; warps 4-7 then convert key
columns [64, 128) to bf16 while the chain converts [0, 64)).  The epilogue waits
obar[b] before reading O1 / O2 (16x256b TMEM loads: 8x8 fragments of value rows
x token pairs) and stores each output tile [token][value] into a K-major SW128
smem tile with stmatrix.trans; a full chunk leaves by TMA (warp 4), a partial
last chunk by 16-byte row stores of its valid tokens.  CTAs without outputs (context end
states / transition rows) skip the O MMAs, the Q / Mqk loads and the epilogue
handshake.

One CTA = one (sequence or context, head), 128 value rows; thread v <-> TMEM
lane v.  Same math and prep as gdn_recur:

    U^T = S^T K^T,  V'^T = V^T A1^T + (-e U^T) A1^T,
    O^T = scale e (S^T Q^T) + V'^T Mqk^T,  S^T = e_C S^T + (f V'^T) K.

TMEM columns: S^T [0,128), U^T [128,192) (then W bf16 in [128,160)),
O1/O2 double buffered [192,448) (buffer b: O1 [192 + 128 b, +64), O2 [256 + 128 b,
+64); the bf16 state copy read by U / O1 of chunk l lives in the O2 half of buffer
l % 2 until O2 of chunk l, which waits mbs, overwrites it), V'^T [448,512) (then
fV' bf16 in [448,480) and V' bf16 in [480,512)).  Lazy decay: S_true = lam * S_tmem (lam *= e_C per chunk; when
the next lam would drop below 1e-12 the chain rescales S_tmem by it before the
update and resets lam = 1), folded into the per-token tables.

MODE 0: whole sequences (grid (Hv, N)); optional device-side two-sequence pick.
MODE 1: context pass 1, z = 2 * context slot + kind: kind 0 = context 0 with
outputs (from h0) or the end state of a later non-final context from zero (HB);
kind 1 = transition rows from the identity with V = 0 (XB).  A non-output
context whose log2 decay sum is < -34 with max beta |k|^2 <= 2 has a transition
of norm <= 2^-34: its kind 1 CTA only clears XF, and its kind 0 CTA starts at the
shortest trailing window whose decay sum is < -34.
MODE 2: contexts >= 1 from the combined start states SB, with outputs.
"""

import cutlass
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.experimental.primitives.nvvm_wrapper import (
    MMALayout,
    cp_async_bulk_shared_cluster_global,
    stmatrix,
    tcgen05_fence,
)

C = 64
D = 128
FAC = 192
CTRI = 2560
NW = 10
NR = 8
NT = 32 * NW
NB = 288
BAR_S = 3
BAR_W = 5
BAR_P = 6
BAR_R = 7
BAR_E = 4


def kmajor(rows, cols, stages=None):
    atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16)
    if stages is None:
        return cute.tile_to_shape(atom, (rows, cols), (0, 1))
    return cute.tile_to_shape(atom, (rows, cols, stages), (0, 1, 2))


def mnmajor_kt(stages):
    """(N = key 128, K = token 64) MN-major view with the physical layout of kmajor(64, 128)."""
    atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16)
    return cute.tile_to_shape(atom, (D, C, stages), (1, 0, 2))


@cute.struct
class Bars:
    mbu: cute.struct.MemRange[cutlass.Int64, 1]
    mbv: cute.struct.MemRange[cutlass.Int64, 1]
    mbs: cute.struct.MemRange[cutlass.Int64, 1]
    obar: cute.struct.MemRange[cutlass.Int64, 2]
    ofree: cute.struct.MemRange[cutlass.Int64, 2]
    epb: cute.struct.MemRange[cutlass.Int64, 2]
    full: cute.struct.MemRange[cutlass.Int64, 3]
    sfree: cute.struct.MemRange[cutlass.Int64, 3]
    sdone: cute.struct.MemRange[cutlass.Int64, 1]
    tmem_slot: cutlass.Int32


def tma_view(sT: cute.Tensor):
    """(key, 1, token) view of a K-major (token, key) smem tile, as the TMA atom expects."""
    return cute.make_tensor(
        sT.iterator,
        cute.make_layout(
            (sT.shape[1], 1, sT.shape[0]),
            stride=(sT.layout.stride[1], 0, sT.layout.stride[0]),
        ),
    )


@cute.jit
def tma_tri(mat, st, sT, bar, Atoms, Maps):
    """Compact lower-triangular 64x64 tile `mat` -> stage st of sT: row block r (16 rows)
    from its (16 (r + 1))-column global view; TMA zero-fills the columns past it."""
    one = cute.make_layout(1)
    for r in cutlass.range_constexpr(4):
        g_tile = cute.local_tile(cute.domain_offset((0, mat, 0), Maps[r]), (C, 1, 16), (0, 0, 0))
        s_blk = cute.local_tile(sT[None, None, st], (16, C), (r, 0))
        dd, ss = cpasync.tma_partition(
            Atoms[r],
            0,
            one,
            cute.group_modes(tma_view(s_blk), 0, 3),
            cute.group_modes(g_tile, 0, 3),
        )
        cute.copy(Atoms[r], ss, dd, tma_bar_ptr=bar)


@cute.jit
def tma_chunk(
    cc,
    st,
    bos,
    slot0,
    hq,
    h,
    HV,
    FC,
    sK,
    sQ,
    sV,
    sA,
    sM,
    sF,
    bar,
    KAtom: cute.CopyAtom,
    KMap: cute.Tensor,
    QAtom: cute.CopyAtom,
    QMap: cute.Tensor,
    VAtom: cute.CopyAtom,
    VMap: cute.Tensor,
    AAtoms,
    AMaps,
    MAtoms,
    MMaps,
    load_v,
    load_qm,
):
    """Loader warp: TMA loads of chunk cc (rows past the sequence end are other tokens or
    zero fill; the prep zeroes their A1 rows / columns and Mqk columns) into stage st."""
    start = bos + cc * C
    slot = slot0 + cc
    nbytes = C * D * 2 + C * C * 2 + FAC * 4
    if load_qm == 1:
        nbytes = nbytes + C * D * 2 + C * C * 2
    if load_v == 1:
        nbytes = nbytes + C * D * 2
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(bar, nbytes)
    one = cute.make_layout(1)
    k_tile = cute.local_tile(cute.domain_offset((0, hq, start), KMap), (D, 1, C), (0, 0, 0))
    kd, ks = cpasync.tma_partition(
        KAtom,
        0,
        one,
        cute.group_modes(tma_view(sK[None, None, st]), 0, 3),
        cute.group_modes(k_tile, 0, 3),
    )
    cute.copy(KAtom, ks, kd, tma_bar_ptr=bar)
    tma_tri(slot * HV + h, st, sA, bar, AAtoms, AMaps)
    if load_qm == 1:
        q_tile = cute.local_tile(cute.domain_offset((0, hq, start), QMap), (D, 1, C), (0, 0, 0))
        qd, qs = cpasync.tma_partition(
            QAtom,
            0,
            one,
            cute.group_modes(tma_view(sQ[None, None, st]), 0, 3),
            cute.group_modes(q_tile, 0, 3),
        )
        cute.copy(QAtom, qs, qd, tma_bar_ptr=bar)
        tma_tri(slot * HV + h, st, sM, bar, MAtoms, MMaps)
    if load_v == 1:
        v_tile = cute.local_tile(cute.domain_offset((0, h, start), VMap), (D, 1, C), (0, 0, 0))
        vd, vs = cpasync.tma_partition(
            VAtom,
            0,
            one,
            cute.group_modes(tma_view(sV[None, None, st]), 0, 3),
            cute.group_modes(v_tile, 0, 3),
        )
        cute.copy(VAtom, vs, vd, tma_bar_ptr=bar)
    with cute.arch.elect_one():
        cp_async_bulk_shared_cluster_global(
            sF.iterator + st * FAC,
            FC.iterator + (slot * HV + h) * FAC,
            bar,
            FAC * 4,
        )


@cute.kernel
def tc_kernel(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    A1: cute.Tensor,
    MQK: cute.Tensor,
    FC: cute.Tensor,
    H0: cute.Tensor,
    Cu: cute.Tensor,
    O: cute.Tensor,
    FIN: cute.Tensor,
    HB: cute.Tensor,
    XB: cute.Tensor,
    SB: cute.Tensor,
    XF: cute.Tensor,
    nseq: cutlass.Int32,
    HQ: cutlass.Int32,
    HV: cutlass.Int32,
    dense_t: cutlass.Int32,
    lc: cutlass.Int32,
    scale: cutlass.Float32,
    ratio: cutlass.Int32,
    mma64: cute.TiledMma,
    mma64v: cute.TiledMma,
    mma64t: cute.TiledMma,
    mma64tm: cute.TiledMma,
    mma32: cute.TiledMma,
    copy_kq: cute.TiledCopy,
    copy_v: cute.TiledCopy,
    copy_a: cute.TiledCopy,
    kq_layout: cute.ComposedLayout,
    kt_layout: cute.ComposedLayout,
    a_layout: cute.ComposedLayout,
    o_layout: cute.ComposedLayout,
    KAtom: cute.CopyAtom,
    KMap: cute.Tensor,
    QAtom: cute.CopyAtom,
    QMap: cute.Tensor,
    VAtom: cute.CopyAtom,
    VMap: cute.Tensor,
    AAtom0: cute.CopyAtom,
    AMap0: cute.Tensor,
    AAtom1: cute.CopyAtom,
    AMap1: cute.Tensor,
    AAtom2: cute.CopyAtom,
    AMap2: cute.Tensor,
    AAtom3: cute.CopyAtom,
    AMap3: cute.Tensor,
    MAtom0: cute.CopyAtom,
    MMap0: cute.Tensor,
    MAtom1: cute.CopyAtom,
    MMap1: cute.Tensor,
    MAtom2: cute.CopyAtom,
    MMap2: cute.Tensor,
    MAtom3: cute.CopyAtom,
    MMap3: cute.Tensor,
    OAtom: cute.CopyAtom,
    OMap: cute.Tensor,
    STAGES: cutlass.Constexpr,
    HAS_H: cutlass.Constexpr,
    PICK: cutlass.Constexpr,
    MODE: cutlass.Constexpr,
):
    gtid, _, _ = cute.arch.thread_idx()
    AAtom = [AAtom0, AAtom1, AAtom2, AAtom3]
    AMap = [AMap0, AMap1, AMap2, AMap3]
    MAtom = [MAtom0, MAtom1, MAtom2, MAtom3]
    MMap = [MMap0, MMap1, MMap2, MMap3]
    if cutlass.const_expr(MODE != 0 or PICK != 0):
        cute.arch.griddepcontrol_launch_dependents()
    tid = gtid % 128
    lane = gtid % 32
    h, zz, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    hq = h // (HV // HQ)

    kind = cutlass.Int32(0)
    cslot = zz
    if cutlass.const_expr(MODE == 1):
        kind = zz % 2
        cslot = zz // 2

    run = cutlass.Int32(1)
    if cutlass.const_expr(PICK != 0):
        u0 = Cu[0]
        u1 = Cu[1]
        u2 = Cu[2]
        l0 = u1 - u0
        l1 = u2 - u1
        near = cutlass.Int32(0)
        if 2 * l0 <= ratio * l1:
            if 2 * l1 <= ratio * l0:
                near = 1
        if cutlass.const_expr(PICK == 1):
            if near == 1:
                run = 0
        else:
            if near == 0:
                run = 0
    seq = cutlass.Int32(0)
    bos = cutlass.Int32(0)
    eos = cutlass.Int32(0)
    if run == 1:
        if cutlass.const_expr(MODE == 0):
            seq = zz
        else:
            for i in range(nseq):
                b_i = i * dense_t
                if dense_t == 0:
                    b_i = Cu[i]
                if (b_i // C) // lc + i <= cslot:
                    seq = i
        bos = seq * dense_t
        eos = bos + dense_t
        if dense_t == 0:
            bos = Cu[seq]
            eos = Cu[seq + 1]
    nch_all = (eos - bos + C - 1) // C
    slot0 = bos // C + seq
    c0 = cutlass.Int32(0)
    c1 = nch_all
    nctx = cutlass.Int32(1)
    jctx = cutlass.Int32(0)
    active = cutlass.Int32(1)
    do_out = cutlass.Int32(1)
    if cutlass.const_expr(MODE != 0):
        jctx = cslot - ((bos // C) // lc + seq)
        nctx = (nch_all + lc - 1) // lc
        c0 = jctx * lc
        c1 = c0 + lc
        c1 = min(c1, nch_all)
        if cutlass.const_expr(MODE == 1):
            if kind == 0:
                do_out = 0
                if jctx == 0:
                    do_out = 1
                if jctx > nctx - 2:
                    active = 0
                if jctx == 0:
                    active = 1
            else:
                do_out = 0
                if jctx < 1:
                    active = 0
                if jctx > nctx - 2:
                    active = 0
        else:
            if jctx < 1:
                active = 0
            if jctx >= nctx:
                active = 0
        if c0 >= nch_all:
            active = 0
    if run == 0:
        active = 0

    smem = cutlass.utils.SmemAllocator()
    bars = smem.allocate(Bars)
    sRed = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * NR + 1), 16)
    sK = smem.allocate_tensor(cutlass.BFloat16, kq_layout.outer, byte_alignment=1024, swizzle=kq_layout.inner)
    sQ = smem.allocate_tensor(cutlass.BFloat16, kq_layout.outer, byte_alignment=1024, swizzle=kq_layout.inner)
    sV = smem.allocate_tensor(cutlass.BFloat16, kq_layout.outer, byte_alignment=1024, swizzle=kq_layout.inner)
    sEp = smem.allocate_tensor(cutlass.Float32, cute.make_layout((C, 2), stride=(1, C)), 16)
    sWe = smem.allocate_tensor(cutlass.Float32, cute.make_layout(C), 16)
    sFi = smem.allocate_tensor(cutlass.Float32, cute.make_layout(C), 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
    sM = smem.allocate_tensor(cutlass.BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
    sF = smem.allocate_tensor(cutlass.Float32, cute.make_layout((FAC, STAGES), stride=(1, FAC)), 16)

    sOut = smem.allocate_tensor(cutlass.BFloat16, o_layout.outer, byte_alignment=1024, swizzle=o_layout.inner)
    sOutT = tma_view(sOut)

    sKt = cute.make_tensor(sK.iterator, kt_layout.outer)
    sVt = cute.make_tensor(sV.iterator, kt_layout.outer)

    if cutlass.const_expr(MODE == 1):
        if active == 1:
            if do_out == 0:
                if warp < NR:
                    pg = cutlass.Float32(0.0)
                    pm = cutlass.Float32(0.0)
                    for cc in range(c0 + gtid, c1, 32 * NR):
                        fb = ((slot0 + cc) * HV + h) * FAC
                        pg = pg + FC[fb + 2 * C + 2]
                        pm = cute.arch.fmax(pm, FC[fb + 2 * C + 1])
                    pg = cute.arch.warp_reduction_sum(pg)
                    pm = cute.arch.warp_reduction_max(pm)
                    if lane == 0:
                        sRed[warp] = pg
                        sRed[NR + warp] = pm
                cute.arch.sync_threads()
                tg = cutlass.Float32(0.0)
                tm = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(NR):
                    tg = tg + sRed[w]
                    tm = cute.arch.fmax(tm, sRed[NR + w])
                trunc = cutlass.Int32(0)
                if tg < -34.0:
                    if tm <= 2.0:
                        trunc = 1
                if kind == 1:
                    if trunc == 1:
                        active = 0
                    if gtid == 0:
                        XF[cslot * HV + h] = 1 - trunc
                else:
                    if trunc == 1:
                        if warp == 0:
                            nctx_ch = c1 - c0
                            carry = cutlass.Float32(0.0)
                            first = cutlass.Int32(-1)
                            for b in range((nctx_ch + 31) // 32):
                                if first < 0:
                                    t = b * 32 + lane
                                    val = cutlass.Float32(0.0)
                                    if t < nctx_ch:
                                        val = FC[((slot0 + c1 - 1 - t) * HV + h) * FAC + 2 * C + 2]
                                    incl = val
                                    for off in cutlass.range_constexpr(5):
                                        other = cute.arch.shuffle_sync_up(incl, 1 << off)
                                        if lane >= (1 << off):
                                            incl = incl + other
                                    hit = cutlass.Int32(0)
                                    if t < nctx_ch:
                                        if carry + incl < -34.0:
                                            hit = 1
                                    bal = cute.arch.vote_ballot_sync(hit == 1)
                                    if bal != 0:
                                        first = b * 32 + cute.arch.popc((bal - 1) & (bal ^ -1))
                                    carry = carry + cute.arch.shuffle_sync(incl, 31)
                            if lane == 0:
                                sRed[2 * NR] = cutlass.Float32(first + 1)
                        cute.arch.sync_threads()
                        win = cutlass.Int32(sRed[2 * NR])
                        if win > 0:
                            c0 = c1 - win
    if active == 1:
        nloc = c1 - c0
        load_v = cutlass.Int32(1)
        if cutlass.const_expr(MODE == 1):
            if kind == 1:
                load_v = 0
        mbu = bars.mbu.data_ptr()
        mbv = bars.mbv.data_ptr()
        mbs = bars.mbs.data_ptr()
        obar = bars.obar.data_ptr()
        ofree = bars.ofree.data_ptr()
        epb = bars.epb.data_ptr()
        full = bars.full.data_ptr()
        sfree = bars.sfree.data_ptr()
        sdone = bars.sdone.data_ptr()
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbu, 1)
                cute.arch.mbarrier_init(mbv, 1)
                cute.arch.mbarrier_init(mbs, 1)
                cute.arch.mbarrier_init(sdone, 1)
                for b in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(obar + b, 1)
                    cute.arch.mbarrier_init(ofree + b, 128)
                    cute.arch.mbarrier_init(epb + b, 1)
                for b in cutlass.range_constexpr(STAGES):
                    cute.arch.mbarrier_init(full + b, 1)
                    cute.arch.mbarrier_init(sfree + b, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        tmem = utils.TmemAllocator(
            bars.tmem_slot.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=NT),
        )
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tptr = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()

        thr64 = mma64.get_slice(0)
        acc64_layout = mma64.make_fragment_C(mma64.partition_shape_C((D, C))).layout
        tU = cute.make_tensor(tptr + 128, acc64_layout)
        tV = cute.make_tensor(tptr + 448, acc64_layout)
        tO_a = cute.make_tensor(tptr + 192, acc64_layout)
        tO2_a = cute.make_tensor(tptr + 256, acc64_layout)
        tO_b = cute.make_tensor(tptr + 320, acc64_layout)
        tO2_b = cute.make_tensor(tptr + 384, acc64_layout)

        acc32_layout = mma32.make_fragment_C(mma32.partition_shape_C((D, 32))).layout
        tW32 = cute.make_tensor(tptr + 128, acc32_layout)
        st32 = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Float32),
            tW32,
        )
        st32_t = st32.get_slice(tid)
        tW_dst = st32_t.partition_D(tW32)

        tP_dst = st32_t.partition_D(cute.make_tensor(tptr + 448, acc32_layout))
        tR_dst = st32_t.partition_D(cute.make_tensor(tptr + 480, acc32_layout))

        tSb_dst00 = st32_t.partition_D(cute.make_tensor(tptr + 256, acc32_layout))
        tSb_dst01 = st32_t.partition_D(cute.make_tensor(tptr + 288, acc32_layout))
        tSb_dst10 = st32_t.partition_D(cute.make_tensor(tptr + 384, acc32_layout))
        tSb_dst11 = st32_t.partition_D(cute.make_tensor(tptr + 416, acc32_layout))
        ld32 = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32),
            tW32,
        )
        thr32 = mma32.get_slice(0)
        c32 = ld32.get_slice(tid).partition_D(thr32.partition_C(cute.make_identity_tensor((D, 32))))
        rsb = cute.make_rmem_tensor(c32.shape, cutlass.Float32)

        ld64 = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32),
            tU,
        )
        ld64_t = ld64.get_slice(tid)
        st64 = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x64), cutlass.Float32),
            tU,
        )
        st64_t = st64.get_slice(tid)
        c64 = ld64_t.partition_D(thr64.partition_C(cute.make_identity_tensor((D, C))))
        rU = cute.make_rmem_tensor(c64.shape, cutlass.Float32)
        tU_src = ld64_t.partition_S(tU)
        tV_src = ld64_t.partition_S(tV)
        rO2 = cute.make_rmem_tensor(c64.shape, cutlass.Float32)

        tS_lo = cute.make_tensor(tptr, acc64_layout)
        tS_hi = cute.make_tensor(tptr + 64, acc64_layout)
        tSlo_src = ld64_t.partition_S(tS_lo)
        tShi_src = ld64_t.partition_S(tS_hi)
        tSlo_dst = st64_t.partition_D(tS_lo)
        tShi_dst = st64_t.partition_D(tS_hi)

        hbase = (seq * HV + h) * (D * D)
        xbase = (cslot * HV + h) * (D * D)
        vrow = c64[0][0]
        sbf = cute.make_rmem_tensor((C,), cutlass.BFloat16)
        if warp < 4:
            keep = cutlass.Int32(0)
            if cutlass.const_expr(MODE == 2):
                if jctx >= 2:
                    keep = XF[(cslot - 1) * HV + h]
            if cutlass.const_expr(MODE == 1):
                if kind == 1:
                    for st0 in cutlass.range_constexpr(STAGES):
                        for e in range(tid, C * D, 128):
                            sV[e // D, e % D, st0] = cutlass.BFloat16(0.0)
            if cutlass.const_expr(MODE == 2):
                cute.arch.griddepcontrol_wait()
            for hf in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(C):
                    rU[i] = 0.0
                if cutlass.const_expr(MODE == 2):
                    if keep == 0:
                        for i in cutlass.range_constexpr(C):
                            rU[i] = HB[xbase - HV * (D * D) + vrow * D + hf * C + c64[i][1]]
                    else:
                        for i in cutlass.range_constexpr(C):
                            rU[i] = SB[xbase + vrow * D + hf * C + c64[i][1]]
                else:
                    if cutlass.const_expr(HAS_H):
                        if jctx == 0:
                            if kind == 0:
                                for i in cutlass.range_constexpr(C):
                                    rU[i] = H0[hbase + (hf * C + c64[i][1]) * D + vrow]
                    if cutlass.const_expr(MODE == 1):
                        if kind == 1:
                            for i in cutlass.range_constexpr(C):
                                if vrow == hf * C + c64[i][1]:
                                    rU[i] = 1.0
                if cutlass.const_expr(hf == 0):
                    cute.copy(st64, rU, tSlo_dst)
                else:
                    cute.copy(st64, rU, tShi_dst)
                for i in cutlass.range_constexpr(C):
                    sbf[i] = rU[i].to(cutlass.BFloat16)
                pkb = cute.recast_tensor(sbf, cutlass.Float32)
                for j in cutlass.range_constexpr(C // 2):
                    rsb[j] = pkb[j]
                if cutlass.const_expr(hf == 0):
                    cute.copy(st32, rsb, tSb_dst00)
                else:
                    cute.copy(st32, rsb, tSb_dst01)
            cute.arch.fence_view_async_tmem_store()
            cute.arch.fence_proxy("async.shared", space="cta")
            tcgen05_fence("before_thread_sync")
            cute.arch.barrier_arrive(barrier_id=BAR_S, number_of_threads=NB)

            lam = cutlass.Float32(1.0)
            for cl in range(nloc):
                st = cl % STAGES
                ob = cl % 2
                par = cl % 2

                cute.arch.mbarrier_wait(full + st, (cl // STAGES) % 2)

                lnext = lam * sF[2 * C, st]
                lres = lnext
                if lres < 1.0e-12:
                    lnext = cutlass.Float32(1.0)

                if tid < C:
                    e_t = sF[tid, st]
                    sWe[tid] = -(e_t * lam)
                    sEp[tid, ob] = e_t * (scale * lam)
                else:
                    sFi[tid - C] = sF[tid, st] / lnext
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                if do_out == 1:
                    if tid == 0:
                        cute.arch.mbarrier_arrive(epb + ob)
                cute.arch.mbarrier_wait(mbu, par)
                if lres < 1.0e-12:
                    cute.copy(ld64, tSlo_src, rU)
                    for i in cutlass.range_constexpr(C):
                        rU[i] = rU[i] * lres
                    cute.copy(st64, rU, tSlo_dst)
                    cute.copy(ld64, tShi_src, rU)
                    for i in cutlass.range_constexpr(C):
                        rU[i] = rU[i] * lres
                    cute.copy(st64, rU, tShi_dst)
                    cute.arch.fence_view_async_tmem_store()

                cute.copy(ld64, tU_src, rU)
                rr = cute.make_rmem_tensor((C,), cutlass.BFloat16)
                for i in cutlass.range_constexpr(C):
                    t = c64[i][1]
                    rr[i] = (rU[i] * sWe[t]).to(cutlass.BFloat16)
                rpk = cute.recast_tensor(rr, cutlass.Float32)
                rw = cute.make_rmem_tensor(c32.shape, cutlass.Float32)
                for j in cutlass.range_constexpr(C // 2):
                    rw[j] = rpk[j]
                cute.copy(st32, rw, tW_dst)
                cute.arch.fence_view_async_tmem_store()
                tcgen05_fence("before_thread_sync")
                cute.arch.barrier_arrive(barrier_id=BAR_W, number_of_threads=160)
                cute.arch.mbarrier_wait(mbv, par)

                lam = lnext
                cute.copy(ld64, tV_src, rU)

                rpf = cute.make_rmem_tensor((C,), cutlass.BFloat16)
                for i in cutlass.range_constexpr(C):
                    t = c64[i][1]
                    rpf[i] = (rU[i] * sFi[t]).to(cutlass.BFloat16)
                rpk = cute.recast_tensor(rpf, cutlass.Float32)
                rw = cute.make_rmem_tensor(c32.shape, cutlass.Float32)
                for j in cutlass.range_constexpr(C // 2):
                    rw[j] = rpk[j]
                cute.copy(st32, rw, tP_dst)
                cute.arch.fence_view_async_tmem_store()
                tcgen05_fence("before_thread_sync")
                cute.arch.barrier_arrive(barrier_id=BAR_P, number_of_threads=160)
                if do_out == 1:
                    rp = cute.make_rmem_tensor((C,), cutlass.BFloat16)
                    for i in cutlass.range_constexpr(C):
                        rp[i] = rU[i].to(cutlass.BFloat16)
                    rpk2 = cute.recast_tensor(rp, cutlass.Float32)
                    for j in cutlass.range_constexpr(C // 2):
                        rw[j] = rpk2[j]
                    cute.copy(st32, rw, tR_dst)
                    cute.arch.fence_view_async_tmem_store()
                    tcgen05_fence("before_thread_sync")
                    cute.arch.barrier_arrive(barrier_id=BAR_R, number_of_threads=160)
                cute.arch.mbarrier_wait(mbs, par)

                if do_out == 1:
                    if cl >= 1:
                        cute.arch.mbarrier_wait(ofree + (1 - ob), ((cl - 1) // 2) % 2)
                cute.copy(ld64, tSlo_src, rU)
                for i in cutlass.range_constexpr(C):
                    sbf[i] = rU[i].to(cutlass.BFloat16)
                pkb = cute.recast_tensor(sbf, cutlass.Float32)
                for j in cutlass.range_constexpr(C // 2):
                    rsb[j] = pkb[j]
                if ob == 0:
                    cute.copy(st32, rsb, tSb_dst10)
                else:
                    cute.copy(st32, rsb, tSb_dst00)
                cute.arch.fence_view_async_tmem_store()
                tcgen05_fence("before_thread_sync")
                cute.arch.barrier_arrive(barrier_id=BAR_S, number_of_threads=NB)

            if nloc > 0:
                cute.arch.mbarrier_wait(sdone, (nloc - 1) % 2)
            to_fin = cutlass.Int32(1)
            if cutlass.const_expr(MODE == 1):
                if nctx > 1:
                    to_fin = 0
                if kind == 1:
                    to_fin = 0
            if cutlass.const_expr(MODE == 2):
                if jctx != nctx - 1:
                    to_fin = 0
            for hf in cutlass.range_constexpr(2):
                if cutlass.const_expr(hf == 0):
                    cute.copy(ld64, tSlo_src, rU)
                else:
                    cute.copy(ld64, tShi_src, rU)
                if to_fin == 1:
                    for i in cutlass.range_constexpr(C):
                        FIN[hbase + (hf * C + c64[i][1]) * D + vrow] = rU[i] * lam
                if cutlass.const_expr(MODE == 1):
                    if to_fin == 0:
                        if kind == 0:
                            for i in cutlass.range_constexpr(C):
                                HB[xbase + vrow * D + hf * C + c64[i][1]] = rU[i] * lam
                        else:
                            for i in cutlass.range_constexpr(C):
                                XB[xbase + vrow * D + hf * C + c64[i][1]] = rU[i] * lam
        elif warp < 8:
            ld256 = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), cutlass.Float32),
                tO_a,
            )
            ld256_t = ld256.get_slice(tid)
            o1s_a = ld256_t.partition_S(tO_a)
            o2s_a = ld256_t.partition_S(tO2_a)
            o1s_b = ld256_t.partition_S(tO_b)
            o2s_b = ld256_t.partition_S(tO2_b)
            c256 = ld256_t.partition_D(thr64.partition_C(cute.make_identity_tensor((D, C))))
            rA = cute.make_rmem_tensor(c256.shape, cutlass.Float32)
            rB = cute.make_rmem_tensor(c256.shape, cutlass.Float32)
            q4 = lane % 4

            km = lane // 8
            v0 = 32 * (warp % 4) + 16 * (km // 2) + 8 * (km % 2)
            sm_base = sOut.iterator.toint() + (v0 // 64) * (C * 128) + (lane % 8) * 128 + (((v0 % 64) // 8) ^ (lane % 8)) * 16
            rO = cute.make_rmem_tensor((C,), cutlass.BFloat16)
            cute.arch.barrier_arrive(barrier_id=BAR_S, number_of_threads=NB)
            sbh = cute.make_rmem_tensor((C,), cutlass.BFloat16)
            for cl in range(nloc):
                ob = cl % 2
                par = (cl // 2) % 2

                cute.arch.mbarrier_wait(sdone, cl % 2)
                cute.copy(ld64, tShi_src, rO2)
                for i in cutlass.range_constexpr(C):
                    sbh[i] = rO2[i].to(cutlass.BFloat16)
                pkh = cute.recast_tensor(sbh, cutlass.Float32)
                for j in cutlass.range_constexpr(C // 2):
                    rsb[j] = pkh[j]
                if ob == 0:
                    cute.copy(st32, rsb, tSb_dst11)
                else:
                    cute.copy(st32, rsb, tSb_dst01)
                cute.arch.fence_view_async_tmem_store()
                tcgen05_fence("before_thread_sync")
                cute.arch.barrier_arrive(barrier_id=BAR_S, number_of_threads=NB)
                if do_out == 1:
                    cute.arch.mbarrier_wait(obar + ob, par)
                    cute.arch.mbarrier_wait(epb + ob, par)
                    if ob == 0:
                        cute.copy(ld256, o1s_a, rA)
                        cute.copy(ld256, o2s_a, rB)
                    else:
                        cute.copy(ld256, o1s_b, rA)
                        cute.copy(ld256, o2s_b, rB)
                    for j in cutlass.range_constexpr(8):
                        e0 = sEp[8 * j + 2 * q4, ob]
                        e1 = sEp[8 * j + 2 * q4 + 1, ob]
                        for k in cutlass.range_constexpr(4):
                            r0 = 4 * j + 2 * (k % 2) + 32 * (k // 2)
                            rO[8 * j + 2 * k] = (rA[r0] * e0 + rB[r0]).to(cutlass.BFloat16)
                            rO[8 * j + 2 * k + 1] = (rA[r0 + 1] * e1 + rB[r0 + 1]).to(cutlass.BFloat16)
                    pk = cute.recast_tensor(rO, cutlass.Int32)
                    cute.arch.fence_view_async_tmem_load()
                    tcgen05_fence("before_thread_sync")
                    cute.arch.mbarrier_arrive(ofree + ob)

                    if warp == 4:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.arch.barrier(barrier_id=BAR_E, number_of_threads=128)
                    for j in cutlass.range_constexpr(8):
                        stmatrix(
                            cute.make_ptr(
                                cutlass.BFloat16,
                                sm_base + j * 1024,
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            [pk[4 * j], pk[4 * j + 1], pk[4 * j + 2], pk[4 * j + 3]],
                            MMALayout.COL,
                        )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=BAR_E, number_of_threads=128)
                    start = bos + (c0 + cl) * C
                    length = eos - start
                    if length >= C:
                        if warp == 4:
                            o_tile = cute.local_tile(
                                cute.domain_offset((0, h, start), OMap),
                                (D, 1, C),
                                (0, 0, 0),
                            )
                            od, og = cpasync.tma_partition(
                                OAtom,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(sOutT, 0, 3),
                                cute.group_modes(o_tile, 0, 3),
                            )
                            cute.copy(OAtom, od, og)
                            cute.arch.cp_async_bulk_commit_group()
                    for j in cutlass.range_constexpr(C * D // (128 * 8)):
                        idx = j * 128 + tid
                        r = idx // (D // 8)
                        cc = (idx % (D // 8)) * 8
                        if r < length:
                            if length < C:
                                stg = cute.make_rmem_tensor((8,), cutlass.BFloat16)

                                cute.autovec_copy(
                                    cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.BFloat16,
                                            sOut.iterator.toint()
                                            + (cc // 64) * (C * 128)
                                            + r * 128
                                            + (((cc % 64) // 8) ^ (r % 8)) * 16,
                                            cute.AddressSpace.smem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout(8),
                                    ),
                                    stg,
                                )
                                cute.autovec_copy(
                                    stg,
                                    cute.make_tensor(
                                        O.iterator + cute.assume((start + r) * (HV * D) + h * D + cc, divby=8),
                                        cute.make_layout(8),
                                    ),
                                )
            if warp == 4:
                cute.arch.cp_async_bulk_wait_group(0, read=False)
        elif warp == 8:
            thr64v = mma64v.get_slice(0)
            a_tw = mma64t.make_fragment_A(mma64t.partition_shape_A((D, C)))
            tWa = cute.make_tensor(cute.recast_ptr(tptr + 128, dtype=cutlass.BFloat16), a_tw.layout)
            tRa = cute.make_tensor(cute.recast_ptr(tptr + 480, dtype=cutlass.BFloat16), a_tw.layout)
            a_ts = mma64t.make_fragment_A(mma64t.partition_shape_A((D, D)))
            tSb0 = cute.make_tensor(cute.recast_ptr(tptr + 256, dtype=cutlass.BFloat16), a_ts.layout)
            tSb1 = cute.make_tensor(cute.recast_ptr(tptr + 384, dtype=cutlass.BFloat16), a_ts.layout)
            thr_tm = mma64tm.get_slice(0)
            a_tp = mma64tm.make_fragment_A(mma64tm.partition_shape_A((D, C)))
            tPa = cute.make_tensor(cute.recast_ptr(tptr + 448, dtype=cutlass.BFloat16), a_tp.layout)

            cute.arch.barrier(barrier_id=BAR_S, number_of_threads=NB)
            for cl in range(nloc):
                st = cl % STAGES
                ob = cl % 2
                fb_k = mma64.make_fragment_B(thr64.partition_B(sK[None, None, st]))
                fb_q = mma64.make_fragment_B(thr64.partition_B(sQ[None, None, st]))
                fb_a = mma64.make_fragment_B(thr64.partition_B(sA[None, None, st]))
                fa_vt = mma64v.make_fragment_A(thr64v.partition_A(sVt[None, None, st]))
                fb_av = mma64v.make_fragment_B(thr64v.partition_B(sA[None, None, st]))
                fb_m = mma64.make_fragment_B(thr64.partition_B(sM[None, None, st]))

                fb_kt_lo = mma64tm.make_fragment_B(thr_tm.partition_B(cute.local_tile(sKt[None, None, st], (C, C), (0, 0))))
                fb_kt_hi = mma64tm.make_fragment_B(thr_tm.partition_B(cute.local_tile(sKt[None, None, st], (C, C), (1, 0))))
                tOc = tO_a
                tO2c = tO2_a
                tSbc = tSb0
                if ob == 1:
                    tOc = tO_b
                    tO2c = tO2_b
                    tSbc = tSb1
                cute.arch.mbarrier_wait(full + st, (cl // STAGES) % 2)
                if cl > 0:
                    cute.arch.barrier(barrier_id=BAR_S, number_of_threads=NB)
                tcgen05_fence("after_thread_sync")

                mma64t.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(D // 16):
                    cute.gemm(mma64t, tU, tSbc[None, None, k], fb_k[None, None, k], tU)
                    mma64t.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbu)
                mma64v.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(C // 16):
                    cute.gemm(mma64v, tV, fa_vt[None, None, k], fb_av[None, None, k], tV)
                    mma64v.set(tcgen05.Field.ACCUMULATE, True)

                cute.arch.barrier(barrier_id=BAR_W, number_of_threads=160)
                tcgen05_fence("after_thread_sync")
                mma64t.set(tcgen05.Field.ACCUMULATE, True)
                for k in cutlass.range_constexpr(C // 16):
                    cute.gemm(mma64t, tV, tWa[None, None, k], fb_a[None, None, k], tV)
                with cute.arch.elect_one():
                    tcgen05.commit(mbv)
                if do_out == 1:
                    if cl >= 2:
                        cute.arch.mbarrier_wait(ofree + ob, ((cl - 2) // 2) % 2)
                        tcgen05_fence("after_thread_sync")
                    mma64t.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range_constexpr(D // 16):
                        cute.gemm(mma64t, tOc, tSbc[None, None, k], fb_q[None, None, k], tOc)
                        mma64t.set(tcgen05.Field.ACCUMULATE, True)

                cute.arch.barrier(barrier_id=BAR_P, number_of_threads=160)
                tcgen05_fence("after_thread_sync")
                mma64tm.set(tcgen05.Field.ACCUMULATE, True)
                for k in cutlass.range_constexpr(C // 16):
                    cute.gemm(
                        mma64tm,
                        tS_lo,
                        tPa[None, None, k],
                        fb_kt_lo[None, None, k],
                        tS_lo,
                    )

                with cute.arch.elect_one():
                    tcgen05.commit(mbs)
                for k in cutlass.range_constexpr(C // 16):
                    cute.gemm(
                        mma64tm,
                        tS_hi,
                        tPa[None, None, k],
                        fb_kt_hi[None, None, k],
                        tS_hi,
                    )
                with cute.arch.elect_one():
                    tcgen05.commit(sdone)
                if do_out == 1:
                    cute.arch.mbarrier_wait(mbs, cl % 2)
                    cute.arch.barrier(barrier_id=BAR_R, number_of_threads=160)
                    tcgen05_fence("after_thread_sync")
                    mma64t.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range_constexpr(C // 16):
                        cute.gemm(mma64t, tO2c, tRa[None, None, k], fb_m[None, None, k], tO2c)
                        mma64t.set(tcgen05.Field.ACCUMULATE, True)

                with cute.arch.elect_one():
                    tcgen05.commit(obar + ob)
                    tcgen05.commit(sfree + st)

                cute.arch.mbarrier_wait(obar + ob, (cl // 2) % 2)

            if nloc > 0:
                cute.arch.barrier(barrier_id=BAR_S, number_of_threads=NB)
        else:
            for p in cutlass.range_constexpr(STAGES):
                if p < nloc:
                    tma_chunk(
                        c0 + p,
                        p,
                        bos,
                        slot0,
                        hq,
                        h,
                        HV,
                        FC,
                        sK,
                        sQ,
                        sV,
                        sA,
                        sM,
                        sF,
                        full + p,
                        KAtom,
                        KMap,
                        QAtom,
                        QMap,
                        VAtom,
                        VMap,
                        AAtom,
                        AMap,
                        MAtom,
                        MMap,
                        load_v,
                        do_out,
                    )
            for cl in range(nloc - STAGES):
                cute.arch.mbarrier_wait(sfree + cl % STAGES, (cl // STAGES) % 2)
                if cl + STAGES < nloc:
                    tma_chunk(
                        c0 + cl + STAGES,
                        cl % STAGES,
                        bos,
                        slot0,
                        hq,
                        h,
                        HV,
                        FC,
                        sK,
                        sQ,
                        sV,
                        sA,
                        sM,
                        sF,
                        full + cl % STAGES,
                        KAtom,
                        KMap,
                        QAtom,
                        QMap,
                        VAtom,
                        VMap,
                        AAtom,
                        AMap,
                        MAtom,
                        MMap,
                        load_v,
                        do_out,
                    )
        cute.arch.sync_threads()
        tmem.free(tptr)
    if cutlass.const_expr(MODE == 2 or (MODE == 1 and PICK != 0)):
        if cutlass.const_expr(PICK == 0):
            cute.arch.griddepcontrol_wait()
        else:
            hold = run
            if h == 0:
                if zz == 0:
                    hold = 1
            if hold == 1:
                cute.arch.griddepcontrol_wait()


@cute.jit
def tc_launch(
    pQ: cute.Pointer,
    pK: cute.Pointer,
    pV: cute.Pointer,
    pA1: cute.Pointer,
    pMQK: cute.Pointer,
    pFC: cute.Pointer,
    pH0: cute.Pointer,
    pCu: cute.Pointer,
    pO: cute.Pointer,
    pFIN: cute.Pointer,
    pHB: cute.Pointer,
    pXB: cute.Pointer,
    pSB: cute.Pointer,
    pXF: cute.Pointer,
    ntok: cutlass.Int32,
    nslot: cutlass.Int32,
    nseq: cutlass.Int32,
    ncs: cutlass.Int32,
    lc: cutlass.Int32,
    HQ: cutlass.Int32,
    HV: cutlass.Int32,
    dense_t: cutlass.Int32,
    scale: cutlass.Float32,
    ratio: cutlass.Int32,
    STAGES: cutlass.Constexpr,
    HAS_H: cutlass.Constexpr,
    PICK: cutlass.Constexpr,
    MODE: cutlass.Constexpr,
    stream,
):
    Q = cute.make_tensor(pQ, cute.make_layout(ntok * HQ * D))
    K = cute.make_tensor(pK, cute.make_layout(ntok * HQ * D))
    V = cute.make_tensor(pV, cute.make_layout(ntok * HV * D))
    A1 = cute.make_tensor(pA1, cute.make_layout(nslot * HV * C * C))
    MQK = cute.make_tensor(pMQK, cute.make_layout(nslot * HV * C * C))
    FC = cute.make_tensor(pFC, cute.make_layout(nslot * HV * FAC))
    H0 = cute.make_tensor(pH0, cute.make_layout(nseq * HV * D * D))
    Cu = cute.make_tensor(pCu, cute.make_layout(nseq + 1))
    O = cute.make_tensor(pO, cute.make_layout(ntok * HV * D))
    FIN = cute.make_tensor(pFIN, cute.make_layout(nseq * HV * D * D))
    HB = cute.make_tensor(pHB, cute.make_layout(ncs * HV * D * D))
    XB = cute.make_tensor(pXB, cute.make_layout(ncs * HV * D * D))
    SB = cute.make_tensor(pSB, cute.make_layout(ncs * HV * D * D))
    XF = cute.make_tensor(pXF, cute.make_layout(ncs * HV))
    mma64 = cute.make_tiled_mma(
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
    mma64v = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (128, 64, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.K,
        )
    )
    mma64t = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (128, 64, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
    )
    mma64tm = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (128, 64, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
        )
    )
    mma32 = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (128, 32, 16),
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
    copy_kq = cute.make_tiled_copy_tv(atom, cute.make_layout((8, 16), stride=(16, 1)), cute.make_layout((1, 8)))

    in_layout = kmajor(C, D)
    in_outer = in_layout.outer
    tma_in = cute.make_composed_layout(
        in_layout.inner,
        0,
        cute.make_layout(
            (cute.shape(in_outer, mode=[1]), 1, cute.shape(in_outer, mode=[0])),
            stride=(in_outer.stride[1], 0, in_outer.stride[0]),
        ),
    )
    k_g = cute.make_tensor(pK, cute.make_layout((D, HQ, ntok), stride=(1, D, HQ * D)))
    q_g = cute.make_tensor(pQ, cute.make_layout((D, HQ, ntok), stride=(1, D, HQ * D)))
    v_g = cute.make_tensor(pV, cute.make_layout((D, HV, ntok), stride=(1, D, HV * D)))
    g2s = cpasync.CopyBulkTensorTileG2SOp()
    k_atom, k_map = cpasync.make_tiled_tma_atom(g2s, k_g, tma_in, (D, 1, C))
    q_atom, q_map = cpasync.make_tiled_tma_atom(g2s, q_g, tma_in, (D, 1, C))
    v_atom, v_map = cpasync.make_tiled_tma_atom(g2s, v_g, tma_in, (D, 1, C))

    blk = kmajor(16, C)
    tma_blk = cute.make_composed_layout(
        blk.inner,
        0,
        cute.make_layout(
            (cute.shape(blk.outer, mode=[1]), 1, cute.shape(blk.outer, mode=[0])),
            stride=(blk.outer.stride[1], 0, blk.outer.stride[0]),
        ),
    )
    tri = []
    for pT in (pA1, pMQK):
        for r in cutlass.range_constexpr(4):
            w = 16 * (r + 1)
            t_g = cute.make_tensor(
                pT + 128 * r * (r + 1),
                cute.make_layout((w, nslot * HV, 16), stride=(1, CTRI, w)),
            )
            tri += list(cpasync.make_tiled_tma_atom(g2s, t_g, tma_blk, (C, 1, 16)))
    o_g = cute.make_tensor(pO, cute.make_layout((D, HV, ntok), stride=(1, D, HV * D)))
    o_atom, o_map = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), o_g, tma_in, (D, 1, C))
    copy_v = copy_kq
    copy_a = cute.make_tiled_copy_tv(atom, cute.make_layout((16, 8), stride=(8, 1)), cute.make_layout((1, 8)))
    kq_layout = kmajor(C, D, STAGES)
    kt_layout = mnmajor_kt(STAGES)
    a_layout = kmajor(C, C, STAGES)
    zdim = nseq
    if cutlass.const_expr(MODE == 1):
        zdim = 2 * ncs
    if cutlass.const_expr(MODE == 2):
        zdim = ncs
    tc_kernel(
        Q,
        K,
        V,
        A1,
        MQK,
        FC,
        H0,
        Cu,
        O,
        FIN,
        HB,
        XB,
        SB,
        XF,
        nseq,
        HQ,
        HV,
        dense_t,
        lc,
        scale,
        ratio,
        mma64,
        mma64v,
        mma64t,
        mma64tm,
        mma32,
        copy_kq,
        copy_v,
        copy_a,
        kq_layout,
        kt_layout,
        a_layout,
        kmajor(C, D),
        k_atom,
        k_map,
        q_atom,
        q_map,
        v_atom,
        v_map,
        *tri,
        o_atom,
        o_map,
        STAGES,
        HAS_H,
        PICK,
        MODE,
    ).launch(
        grid=(HV, zdim, 1),
        block=(NT, 1, 1),
        stream=stream,
        use_pdl=MODE == 2 or (MODE == 1 and PICK != 0),
    )
