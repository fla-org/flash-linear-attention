# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

"""Chunked Gated DeltaNet recurrence with warp-level tensor cores (CuTe DSL).

Value columns of the key-first state are independent, so the state is kept
value-major, S^T (16 value rows per warp, fp32 registers), and every MMA has the
value rows as M (register A operand):

    U^T  = S^T K^T                         (16 x 64, k = 128)
    R^T  = V^T - e * U^T                   (e_t = exp(gamma_t), per column)
    V'^T = R^T A1^T                        (A1 = (I+L)^-1 diag(beta))
    O^T  = scale e * (S^T Q^T) + V'^T Mqk^T
    S^T <- e_C S^T + (f * V'^T) K          (f_t = exp(gamma_C - gamma_t))

One CTA = W warps = 16 W value rows of one (sequence, value head); grid
(128 / (16 W), Hv, N).  Chunk operands are double buffered with cp.async.
"""

import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync, warp

C = 64
D = 128
FAC = 192
CXF = 64


def swz_layout(rows, cols, stages=None):
    atom = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1)))
    if stages is None:
        return cute.tile_to_shape(atom, (rows, cols), (0, 1))
    return cute.tile_to_shape(atom, (rows, cols, stages), (0, 1, 2))


def a_view(t):
    """C-fragment (MMA, M, N) of a 16x8 atom -> A-fragment ((MMA, 2), M, N / 2)."""
    d = cute.logical_divide(t.layout, (None, None, 2))
    lay = cute.make_layout(
        ((d.shape[0], d.shape[2][0]), d.shape[1], d.shape[2][1]),
        stride=((d.stride[0], d.stride[2][0]), d.stride[1], d.stride[2][1]),
    )
    return cute.make_tensor(t.iterator, lay)


def pair_view(t, mode):
    """Split mode `mode` (n8 tiles) of a fragment into (2, n/2) pairs."""
    tiler = [None] * cute.rank(t.layout)
    tiler[mode] = 2
    return cute.make_tensor(t.iterator, cute.logical_divide(t.layout, tuple(tiler)))


@cute.jit
def tri_gemm(tiled_mma, ld_b, acc, a, bfrag, s_b, v_b):
    """acc += a @ b^T restricted to the lower block-triangle (k-block <= n-pair)."""
    acc2 = pair_view(acc, 2)
    b2 = pair_view(bfrag, 1)
    for kb in cutlass.range_constexpr(4):
        for i in cutlass.range_constexpr(kb, 4):
            cute.copy(ld_b, s_b[None, i, kb], v_b[None, i, kb])
        for i in cutlass.range_constexpr(kb, 4):
            cute.gemm(
                tiled_mma,
                acc2[None, None, (None, i)],
                a[None, None, kb],
                b2[None, (None, i), kb],
                acc2[None, None, (None, i)],
            )


@cute.jit
def load_chunk(
    cc,
    st,
    tid,
    bos,
    eos,
    slot0,
    hq,
    h,
    vbase,
    HQ,
    HV,
    Q,
    K,
    V,
    A1,
    MQK,
    FC,
    sK,
    sQ,
    sV,
    sA,
    sM,
    sF,
    copy_kq: cute.TiledCopy,
    copy_v: cute.TiledCopy,
    copy_a: cute.TiledCopy,
    W: cutlass.Constexpr,
    load_v,
):
    thr_kq = copy_kq.get_slice(tid)
    thr_v = copy_v.get_slice(tid)
    thr_a = copy_a.get_slice(tid)
    c_kq = thr_kq.partition_S(cute.make_identity_tensor((C, D)))
    c_v = thr_v.partition_S(cute.make_identity_tensor((C, 16 * W)))
    fac_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    start = bos + cc * C
    length = eos - start
    slot = slot0 + cc
    kq_off = start * (HQ * D) + hq * D
    gK = cute.make_tensor(
        K.iterator + cute.assume(kq_off, divby=8),
        cute.make_layout((C, D), stride=(cute.assume(HQ * D, divby=128), 1)),
    )
    gQ = cute.make_tensor(
        Q.iterator + cute.assume(kq_off, divby=8),
        cute.make_layout((C, D), stride=(cute.assume(HQ * D, divby=128), 1)),
    )
    gV = cute.make_tensor(
        V.iterator + cute.assume(start * (HV * D) + h * D + vbase, divby=8),
        cute.make_layout((C, 16 * W), stride=(cute.assume(HV * D, divby=128), 1)),
    )
    a_off = (slot * HV + h) * (C * C)
    gA = cute.make_tensor(
        A1.iterator + cute.assume(a_off, divby=8),
        cute.make_layout((C, C), stride=(C, 1)),
    )
    gM = cute.make_tensor(
        MQK.iterator + cute.assume(a_off, divby=8),
        cute.make_layout((C, C), stride=(C, 1)),
    )
    tKg = thr_kq.partition_S(gK)
    tQg = thr_kq.partition_S(gQ)
    tKs = thr_kq.partition_D(sK[None, None, st])
    tQs = thr_kq.partition_D(sQ[None, None, st])
    for m in cutlass.range_constexpr(cute.size(tKs.shape[1])):
        if c_kq[0, m, 0][0] < length:
            cute.copy(copy_kq, tKg[None, m, None], tKs[None, m, None])
            cute.copy(copy_kq, tQg[None, m, None], tQs[None, m, None])
        else:
            tKs[None, m, None].fill(0)
            tQs[None, m, None].fill(0)
    tVg = thr_v.partition_S(gV)
    tVs = thr_v.partition_D(sV[None, None, st])
    if load_v:
        for m in cutlass.range_constexpr(cute.size(tVs.shape[1])):
            if c_v[0, m, 0][0] < length:
                cute.copy(copy_v, tVg[None, m, None], tVs[None, m, None])
            else:
                tVs[None, m, None].fill(0)
    cute.copy(copy_a, thr_a.partition_S(gA), thr_a.partition_D(sA[None, None, st]))
    cute.copy(copy_a, thr_a.partition_S(gM), thr_a.partition_D(sM[None, None, st]))
    if tid < FAC // 4:
        f_off = (slot * HV + h) * FAC + tid * 4
        src = cute.make_tensor(FC.iterator + cute.assume(f_off, divby=4), cute.make_layout(4))
        dst = cute.make_tensor(
            sF.iterator + cute.assume(st * FAC + tid * 4, divby=4),
            cute.make_layout(4),
        )
        cute.copy(fac_atom, src, dst)


@cute.kernel
def recur_kernel(
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
    tiled_mma: cute.TiledMma,
    copy_kq: cute.TiledCopy,
    copy_v: cute.TiledCopy,
    copy_a: cute.TiledCopy,
    kq_layout: cute.ComposedLayout,
    v_layout: cute.Layout,
    a_layout: cute.ComposedLayout,
    W: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    HAS_H: cutlass.Constexpr,
    MODE: cutlass.Constexpr,
    PICK: cutlass.Constexpr,
    LPT: cutlass.Constexpr,
):

    tid, _, _ = cute.arch.thread_idx()
    vb, h, zz = cute.arch.block_idx()
    lane = tid % 32
    wid = tid // 32
    NT = 32 * W
    hq = h // (HV // HQ)
    vbase = vb * (16 * W)
    vrow = vbase + wid * 16

    kind = cutlass.Int32(0)
    cslot = zz
    if cutlass.const_expr(MODE == 1):
        kind = zz % 2
        cslot = zz // 2
    seq = cutlass.Int32(0)
    jctx = cutlass.Int32(0)
    smem = cutlass.utils.SmemAllocator()
    if cutlass.const_expr(MODE == 0):
        seq = zz
        sSeq = smem.allocate_tensor(cutlass.Int32, cute.make_layout(1), 4)
        if cutlass.const_expr(LPT):
            for i in range(tid, nseq, NT):
                li = Cu[i + 1] - Cu[i]
                rank = cutlass.Int32(0)
                for j in range(nseq):
                    lj = Cu[j + 1] - Cu[j]
                    if lj > li:
                        rank += 1
                    if lj == li:
                        if j < i:
                            rank += 1
                if rank == zz:
                    sSeq[0] = i
            cute.arch.sync_threads()
            seq = sSeq[0]
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
    if cutlass.const_expr(PICK != 0):
        l0 = Cu[1] - Cu[0]
        l1 = Cu[2] - Cu[1]
        near = cutlass.Int32(0)
        if 2 * l0 <= lc * l1:
            if 2 * l1 <= lc * l0:
                near = 1
        if cutlass.const_expr(PICK == 1):
            if near == 1:
                active = 0
        else:
            if near == 0:
                active = 0

    sRed = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * W + 1), 16)
    if cutlass.const_expr(MODE == 1):
        if active == 1:
            if do_out == 0:
                pg = cutlass.Float32(0.0)
                pm = cutlass.Float32(0.0)
                for cc in range(c0 + tid, c1, NT):
                    fb = ((slot0 + cc) * HV + h) * FAC
                    pg = pg + FC[fb + 2 * C + 2]
                    pm = cute.arch.fmax(pm, FC[fb + 2 * C + 1])
                pg = cute.arch.warp_reduction_sum(pg)
                pm = cute.arch.warp_reduction_max(pm)
                if lane == 0:
                    sRed[wid] = pg
                    sRed[W + wid] = pm
                cute.arch.sync_threads()
                tg = cutlass.Float32(0.0)
                tm = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(W):
                    tg = tg + sRed[w]
                    tm = cute.arch.fmax(tm, sRed[W + w])
                trunc = cutlass.Int32(0)
                if tg < -34.0:
                    if tm <= 2.0:
                        trunc = 1
                if kind == 1:
                    if trunc == 1:
                        active = 0
                    if vb == 0:
                        if tid == 0:
                            XF[cslot * HV + h] = 1 - trunc
                else:
                    if trunc == 1:
                        if wid == 0:
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
                                sRed[2 * W] = cutlass.Float32(first + 1)
                        cute.arch.sync_threads()
                        win = cutlass.Int32(sRed[2 * W])
                        if win > 0:
                            c0 = c1 - win
    sK = smem.allocate_tensor(cutlass.BFloat16, kq_layout, byte_alignment=1024)
    sQ = smem.allocate_tensor(cutlass.BFloat16, kq_layout, byte_alignment=1024)
    sV = smem.allocate_tensor(cutlass.BFloat16, v_layout, byte_alignment=1024)
    sA = smem.allocate_tensor(cutlass.BFloat16, a_layout, byte_alignment=1024)
    sM = smem.allocate_tensor(cutlass.BFloat16, a_layout, byte_alignment=1024)
    sF = smem.allocate_tensor(cutlass.Float32, cute.make_layout((FAC, STAGES), stride=(1, FAC)), 16)
    sO = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((16, C, W), stride=(1, 24, 24 * C)),
        16,
    )

    thr_mma = tiled_mma.get_slice(lane)
    ld_b = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16),
        tiled_mma,
    )
    ld_bt = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), cutlass.BFloat16),
        tiled_mma,
    )
    ld_at = cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), cutlass.BFloat16),
        tiled_mma,
    )
    st_ct = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.StMatrix8x8x16bOp(True, 4), cutlass.BFloat16),
        tiled_mma,
    )
    t_ld_b = ld_b.get_slice(lane)
    t_ld_bt = ld_bt.get_slice(lane)
    t_ld_at = ld_at.get_slice(lane)
    t_st_ct = st_ct.get_slice(lane)

    acc_s = cute.make_rmem_tensor(thr_mma.partition_shape_C((16, D)), cutlass.Float32)
    c_s = thr_mma.partition_C(cute.make_identity_tensor((16, D)))
    c_c = thr_mma.partition_C(cute.make_identity_tensor((16, C)))
    hbase = (seq * HV + h) * (D * D)
    xbase = (cslot * HV + h) * (D * D)
    load_v = cutlass.Int32(1)
    for i in cutlass.range_constexpr(cute.size(acc_s)):
        acc_s[i] = 0.0
    if cutlass.const_expr(MODE == 2):
        if active == 1:
            keep = cutlass.Int32(0)
            if jctx >= 2:
                keep = XF[(cslot - 1) * HV + h]
            if keep == 0:
                for i in cutlass.range_constexpr(cute.size(acc_s)):
                    acc_s[i] = HB[xbase - HV * (D * D) + (vrow + c_s[i][0]) * D + c_s[i][1]]
            else:
                for i in cutlass.range_constexpr(cute.size(acc_s)):
                    acc_s[i] = SB[xbase + (vrow + c_s[i][0]) * D + c_s[i][1]]
    else:
        if cutlass.const_expr(HAS_H):
            if jctx == 0:
                if kind == 0:
                    if active == 1:
                        for i in cutlass.range_constexpr(cute.size(acc_s)):
                            acc_s[i] = H0[hbase + c_s[i][1] * D + vrow + c_s[i][0]]
        if cutlass.const_expr(MODE == 1):
            if kind == 1:
                load_v = 0
                for i in cutlass.range_constexpr(cute.size(acc_s)):
                    if vrow + c_s[i][0] == c_s[i][1]:
                        acc_s[i] = 1.0

                for st0 in cutlass.range_constexpr(STAGES):
                    for e in range(tid, C * 16 * W, NT):
                        sV[e % C, e // C, st0] = cutlass.BFloat16(0.0)

    sO_w = sO[None, None, wid]
    tOs = t_st_ct.partition_D(sO_w)
    out_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)

    if active == 0:
        c1 = c0
    for p in cutlass.range_constexpr(STAGES - 1):
        if c0 + p < c1:
            load_chunk(
                c0 + p,
                p,
                tid,
                bos,
                eos,
                slot0,
                hq,
                h,
                vbase,
                HQ,
                HV,
                Q,
                K,
                V,
                A1,
                MQK,
                FC,
                sK,
                sQ,
                sV,
                sA,
                sM,
                sF,
                copy_kq,
                copy_v,
                copy_a,
                W,
                load_v,
            )
        cute.arch.cp_async_commit_group()

    for cl in range(c1 - c0):
        c = c0 + cl
        st = cl % STAGES
        cute.arch.cp_async_wait_group(STAGES - 2)
        cute.arch.sync_threads()
        nxt = cl + STAGES - 1
        if c0 + nxt < c1:
            load_chunk(
                c0 + nxt,
                nxt % STAGES,
                tid,
                bos,
                eos,
                slot0,
                hq,
                h,
                vbase,
                HQ,
                HV,
                Q,
                K,
                V,
                A1,
                MQK,
                FC,
                sK,
                sQ,
                sV,
                sA,
                sM,
                sF,
                copy_kq,
                copy_v,
                copy_a,
                W,
                load_v,
            )
        cute.arch.cp_async_commit_group()

        sK_st = sK[None, None, st]
        sQ_st = sQ[None, None, st]
        sA_st = sA[None, None, st]
        sM_st = sM[None, None, st]
        sKt = cute.composition(sK_st, cute.make_layout((D, C), stride=(C, 1)))
        sV_w = sV[None, (None, wid), st]
        sVt_w = cute.composition(sV_w, cute.make_layout((16, C), stride=(C, 1)))

        r_s = cute.make_fragment_like(acc_s, cutlass.BFloat16)
        r_s.store(acc_s.load().to(cutlass.BFloat16))
        r_sa = a_view(r_s)

        acc_u = cute.make_rmem_tensor(thr_mma.partition_shape_C((16, C)), cutlass.Float32)
        acc_o = cute.make_rmem_tensor(thr_mma.partition_shape_C((16, C)), cutlass.Float32)
        acc_u.fill(0.0)
        acc_o.fill(0.0)
        b_k = thr_mma.make_fragment_B(thr_mma.partition_B(sK_st))
        b_q = thr_mma.make_fragment_B(thr_mma.partition_B(sQ_st))
        s_bk = t_ld_b.partition_S(sK_st)
        s_bq = t_ld_b.partition_S(sQ_st)
        v_bk = t_ld_b.retile(b_k)
        v_bq = t_ld_b.retile(b_q)
        for k in cutlass.range_constexpr(D // 16):
            cute.copy(ld_b, s_bk[None, None, k], v_bk[None, None, k])
            cute.copy(ld_b, s_bq[None, None, k], v_bq[None, None, k])
            cute.gemm(tiled_mma, acc_u, r_sa[None, None, k], b_k[None, None, k], acc_u)
            cute.gemm(tiled_mma, acc_o, r_sa[None, None, k], b_q[None, None, k], acc_o)

        r_v = cute.make_fragment_like(acc_u, cutlass.BFloat16)
        r_va = a_view(r_v)
        cute.copy(ld_at, t_ld_at.partition_S(sVt_w), t_ld_at.retile(r_va))
        r_r = cute.make_fragment_like(acc_u, cutlass.BFloat16)
        for i in cutlass.range_constexpr(cute.size(acc_u)):
            t = c_c[i][1]
            e = sF[t, st]
            r_r[i] = (r_v[i].to(cutlass.Float32) - e * acc_u[i]).to(cutlass.BFloat16)
        r_ra = a_view(r_r)

        acc_p = cute.make_rmem_tensor(thr_mma.partition_shape_C((16, C)), cutlass.Float32)
        acc_p.fill(0.0)
        b_a = thr_mma.make_fragment_B(thr_mma.partition_B(sA_st))
        s_ba = t_ld_b.partition_S(sA_st)
        v_ba = t_ld_b.retile(b_a)
        tri_gemm(tiled_mma, ld_b, acc_p, r_ra, b_a, s_ba, v_ba)

        r_p = cute.make_fragment_like(acc_p, cutlass.BFloat16)
        r_pf = cute.make_fragment_like(acc_p, cutlass.BFloat16)
        for i in cutlass.range_constexpr(cute.size(acc_p)):
            t = c_c[i][1]
            r_p[i] = acc_p[i].to(cutlass.BFloat16)
            r_pf[i] = (acc_p[i] * sF[C + t, st]).to(cutlass.BFloat16)
            acc_o[i] = acc_o[i] * (sF[t, st] * scale)
        r_pa = a_view(r_p)
        r_pfa = a_view(r_pf)

        b_m = thr_mma.make_fragment_B(thr_mma.partition_B(sM_st))
        s_bm = t_ld_b.partition_S(sM_st)
        v_bm = t_ld_b.retile(b_m)
        tri_gemm(tiled_mma, ld_b, acc_o, r_pa, b_m, s_bm, v_bm)

        e_c = sF[2 * C, st]
        for i in cutlass.range_constexpr(cute.size(acc_s)):
            acc_s[i] = acc_s[i] * e_c
        b_kt = thr_mma.make_fragment_B(thr_mma.partition_B(sKt))
        s_bkt = t_ld_bt.partition_S(sKt)
        v_bkt = t_ld_bt.retile(b_kt)
        for k in cutlass.range_constexpr(C // 16):
            cute.copy(ld_bt, s_bkt[None, None, k], v_bkt[None, None, k])
            cute.gemm(tiled_mma, acc_s, r_pfa[None, None, k], b_kt[None, None, k], acc_s)

        r_o = cute.make_fragment_like(acc_o, cutlass.BFloat16)
        r_o.store(acc_o.load().to(cutlass.BFloat16))
        cute.copy(st_ct, t_st_ct.retile(r_o), tOs)
        cute.arch.sync_warp()
        start = bos + c * C
        length = eos - start
        if do_out == 0:
            length = 0
        for j in cutlass.range_constexpr(4):
            idx = lane + 32 * j
            row = idx // 2
            half = idx % 2
            if row < length:
                src = cute.make_tensor(
                    sO_w.iterator + cute.assume(row * 24 + half * 8, divby=8),
                    cute.make_layout(8),
                )
                dst = cute.make_tensor(
                    O.iterator + cute.assume((start + row) * (HV * D) + h * D + vrow + half * 8, divby=8),
                    cute.make_layout(8),
                )
                cute.copy(out_atom, src, dst)
        cute.arch.sync_warp()

    cute.arch.cp_async_wait_group(0)
    if active == 1:
        to_fin = cutlass.Int32(1)
        if cutlass.const_expr(MODE == 1):
            if nctx > 1:
                to_fin = 0
            if kind == 1:
                to_fin = 0
        if cutlass.const_expr(MODE == 2):
            if jctx != nctx - 1:
                to_fin = 0
        if to_fin == 1:
            for i in cutlass.range_constexpr(cute.size(acc_s)):
                FIN[hbase + c_s[i][1] * D + vrow + c_s[i][0]] = acc_s[i]
        if cutlass.const_expr(MODE == 1):
            if to_fin == 0:
                if kind == 0:
                    for i in cutlass.range_constexpr(cute.size(acc_s)):
                        HB[xbase + (vrow + c_s[i][0]) * D + c_s[i][1]] = acc_s[i]
                else:
                    for i in cutlass.range_constexpr(cute.size(acc_s)):
                        XB[xbase + (vrow + c_s[i][0]) * D + c_s[i][1]] = acc_s[i]


@cute.jit
def recur_launch(
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
    W: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    HAS_H: cutlass.Constexpr,
    MODE: cutlass.Constexpr,
    PICK: cutlass.Constexpr,
    LPT: cutlass.Constexpr,
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
    tiled_mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
        (1, 1, 1),
        permutation_mnk=(16, 16, 16),
    )
    atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    NT = 32 * W
    copy_kq = cute.make_tiled_copy_tv(atom, cute.make_layout((NT // 16, 16), stride=(16, 1)), cute.make_layout((1, 8)))
    vch = 16 * W // 8
    copy_v = cute.make_tiled_copy_tv(
        atom,
        cute.make_layout((NT // vch, vch), stride=(vch, 1)),
        cute.make_layout((1, 8)),
    )
    copy_a = cute.make_tiled_copy_tv(atom, cute.make_layout((NT // 8, 8), stride=(8, 1)), cute.make_layout((1, 8)))
    kq_layout = swz_layout(C, D, STAGES)
    v_layout = cute.make_layout((C, (16, W), STAGES), stride=(24, (1, 24 * C), 24 * C * W))
    a_layout = swz_layout(C, C, STAGES)
    zdim = nseq
    if cutlass.const_expr(MODE == 1):
        zdim = 2 * ncs
    if cutlass.const_expr(MODE == 2):
        zdim = ncs
    recur_kernel(
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
        tiled_mma,
        copy_kq,
        copy_v,
        copy_a,
        kq_layout,
        v_layout,
        a_layout,
        W,
        STAGES,
        HAS_H,
        MODE,
        PICK,
        LPT,
    ).launch(grid=(D // (16 * W), HV, zdim), block=(NT, 1, 1), stream=stream)


@cute.kernel
def comb_kernel(
    HB: cute.Tensor,
    XB: cute.Tensor,
    SB: cute.Tensor,
    XF: cute.Tensor,
    Cu: cute.Tensor,
    HV: cutlass.Int32,
    dense_t: cutlass.Int32,
    lc: cutlass.Int32,
    ratio: cutlass.Int32,
    PICK: cutlass.Constexpr,
    CRB_: cutlass.Constexpr,
):
    """K-quarter combine with 2 key columns per thread (dev): thread t owns key columns
    c = t % 64 and c + 64, sums all CRB_ rows over i in 32 q + [0, 32) (q = t // 64) and
    finalizes rows HR q + [0, HR) of both columns from the 4 parts (fixed order).
    """
    KS = 4
    HR = CRB_ // KS
    NT = 2 * D
    KQ = D // KS

    cute.arch.griddepcontrol_launch_dependents()
    tid, _, _ = cute.arch.thread_idx()
    rb, h, seq = cute.arch.block_idx()

    hold = cutlass.Int32(1)
    if cutlass.const_expr(PICK):
        p0 = Cu[1] - Cu[0]
        p1 = Cu[2] - Cu[1]
        if 2 * p0 <= ratio * p1:
            if 2 * p1 <= ratio * p0:
                hold = 0
                if rb == 0:
                    if h == 0:
                        if seq == 0:
                            hold = 1
    if hold == 1:
        cute.arch.griddepcontrol_wait()
    c = tid % (D // 2)
    q = tid // (D // 2)
    kb = q * KQ
    bos = seq * dense_t
    eos = bos + dense_t
    if dense_t == 0:
        bos = Cu[seq]
        eos = Cu[seq + 1]
    nch = (eos - bos + C - 1) // C
    nctx = (nch + lc - 1) // lc
    base = (bos // C) // lc + seq
    smem = cutlass.utils.SmemAllocator()
    sS = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D, CRB_), stride=(CRB_, 1)), 16)
    sX = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D, D, 2), stride=(D, 1, D * D)), 16)
    xatom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    sXF = smem.allocate_tensor(cutlass.Int32, cute.make_layout(CXF), 16)
    sP = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((KS, CRB_, D), stride=(CRB_ * D, D, 1)),
        16,
    )
    a0 = cute.make_rmem_tensor((CRB_,), cutlass.Float32)
    a1 = cute.make_rmem_tensor((CRB_,), cutlass.Float32)
    hv0 = cute.make_rmem_tensor((HR,), cutlass.Float32)
    hv1 = cute.make_rmem_tensor((HR,), cutlass.Float32)
    if cutlass.const_expr(PICK):
        l0 = Cu[1] - Cu[0]
        l1 = Cu[2] - Cu[1]
        if 2 * l0 <= ratio * l1:
            if 2 * l1 <= ratio * l0:
                nctx = 0
    if nctx >= 3:
        for t in range(tid, nctx - 1, NT):
            if t < CXF:
                sXF[t] = XF[(base + t) * HV + h]
        cute.arch.sync_threads()
        if sXF[1] != 0:
            xoff = ((base + 1) * HV + h) * (D * D)
            for it in cutlass.range_constexpr(D * D // (4 * NT)):
                idx = (it * NT + tid) * 4
                src = cute.make_tensor(XB.iterator + cute.assume(xoff + idx, divby=4), cute.make_layout(4))
                dst = cute.make_tensor(sX.iterator + cute.assume(idx, divby=4), cute.make_layout(4))
                cute.copy(xatom, src, dst)
        cute.arch.cp_async_commit_group()
        for j in range(1, nctx - 1):
            slot = base + j
            st = (j - 1) % 2
            fnext = cutlass.Int32(0)
            if j + 1 < nctx - 1:
                if j + 1 < CXF:
                    fnext = sXF[j + 1]
                else:
                    fnext = XF[(slot + 1) * HV + h]
            if j + 1 < nctx - 1:
                if fnext != 0:
                    xoff = ((slot + 1) * HV + h) * (D * D)
                    for it in cutlass.range_constexpr(D * D // (4 * NT)):
                        idx = (it * NT + tid) * 4
                        src = cute.make_tensor(
                            XB.iterator + cute.assume(xoff + idx, divby=4),
                            cute.make_layout(4),
                        )
                        dst = cute.make_tensor(
                            sX.iterator + cute.assume((1 - st) * D * D + idx, divby=4),
                            cute.make_layout(4),
                        )
                        cute.copy(xatom, src, dst)
            cute.arch.cp_async_commit_group()
            fcur = cutlass.Int32(0)
            if j < CXF:
                fcur = sXF[j]
            else:
                fcur = XF[slot * HV + h]
            if fcur != 0:
                pflag = cutlass.Int32(0)
                if j > 1:
                    if j - 1 < CXF:
                        pflag = sXF[j - 1]
                    else:
                        pflag = XF[(slot - 1) * HV + h]
                if pflag == 0:
                    hp = ((slot - 1) * HV + h) * (D * D) + (rb * CRB_ + q * HR) * D
                    for r in cutlass.range_constexpr(HR):
                        sS[c, q * HR + r] = HB[hp + r * D + c]
                        sS[c + D // 2, q * HR + r] = HB[hp + r * D + c + D // 2]
                hb = (slot * HV + h) * (D * D) + (rb * CRB_ + q * HR) * D
                for r in cutlass.range_constexpr(HR):
                    hv0[r] = HB[hb + r * D + c]
                    hv1[r] = HB[hb + r * D + c + D // 2]
                for r in cutlass.range_constexpr(CRB_):
                    a0[r] = cutlass.Float32(0.0)
                    a1[r] = cutlass.Float32(0.0)
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()
                for i0 in range(0, KQ, 8):
                    for u in cutlass.range_constexpr(8):
                        x0 = sX[kb + i0 + u, c, st]
                        x1 = sX[kb + i0 + u, c + D // 2, st]
                        srow = cute.make_tensor(
                            sS.iterator + cute.assume((kb + i0 + u) * CRB_, divby=4),
                            cute.make_layout(CRB_),
                        ).load()
                        for r in cutlass.range_constexpr(CRB_):
                            a0[r] = a0[r] + srow[r] * x0
                            a1[r] = a1[r] + srow[r] * x1
                for r in cutlass.range_constexpr(CRB_):
                    sP[q, r, c] = a0[r]
                    sP[q, r, c + D // 2] = a1[r]
                cute.arch.sync_threads()
                sb = ((slot + 1) * HV + h) * (D * D) + (rb * CRB_ + q * HR) * D
                for r in cutlass.range_constexpr(HR):
                    v0 = hv0[r]
                    v1 = hv1[r]
                    for p in cutlass.range_constexpr(KS):
                        v0 = v0 + sP[p, q * HR + r, c]
                        v1 = v1 + sP[p, q * HR + r, c + D // 2]
                    sS[c, q * HR + r] = v0
                    sS[c + D // 2, q * HR + r] = v1
                    SB[sb + r * D + c] = v0
                    SB[sb + r * D + c + D // 2] = v1
        cute.arch.cp_async_wait_group(0)


@cute.jit
def comb_launch(
    pHB: cute.Pointer,
    pXB: cute.Pointer,
    pSB: cute.Pointer,
    pXF: cute.Pointer,
    pCu: cute.Pointer,
    nseq: cutlass.Int32,
    ncs: cutlass.Int32,
    HV: cutlass.Int32,
    dense_t: cutlass.Int32,
    lc: cutlass.Int32,
    ratio: cutlass.Int32,
    PICK: cutlass.Constexpr,
    CRB_: cutlass.Constexpr,
    stream,
):
    HB = cute.make_tensor(pHB, cute.make_layout(ncs * HV * D * D))
    XB = cute.make_tensor(pXB, cute.make_layout(ncs * HV * D * D))
    SB = cute.make_tensor(pSB, cute.make_layout(ncs * HV * D * D))
    XF = cute.make_tensor(pXF, cute.make_layout(ncs * HV))
    Cu = cute.make_tensor(pCu, cute.make_layout(nseq + 1))
    comb_kernel(HB, XB, SB, XF, Cu, HV, dense_t, lc, ratio, PICK, CRB_).launch(
        grid=(D // CRB_, HV, nseq), block=(2 * D, 1, 1), stream=stream, use_pdl=True
    )
