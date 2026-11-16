# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.triadic_kda.chunk_fwd import (
    CHUNK_SIZE,
    MAX_ANCHOR_SPAN,
    MAX_REG_E,
    chunk_position,
    diag_score_tile,
    fwd_h,
    fwd_intra,
    get_state_block_size,
    sequence_bounds,
    state_dot,
)
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_offsets


@triton.jit
def _bwd_read_slice(hp, e, pe, last_pe, mt, g2, k2, kg_t):
    ge = tl.load(g2 + pe + e, mt, other=0.0)
    gel = tl.load(g2 + last_pe + e)
    ke = tl.load(k2 + pe + e, mt, other=0.0) * tl.exp(tl.minimum(gel - ge, 0.0))
    return tl.dot(hp, kg_t, out_dtype=tl.float32) * ke[None, :]


@triton.jit
def _bwd_update_slice(h_prev, e, pe, last_pe, mt, g2, q2, k2, b_beta, gl, dout, df, qg, rg, scale):
    ge = tl.load(g2 + pe + e, mt, other=0.0)
    gel = tl.load(g2 + last_pe + e)
    exp_ge = tl.exp(ge)
    qe = tl.load(q2 + pe + e, mt, other=0.0) * exp_ge
    re = tl.load(k2 + pe + e, mt, other=0.0) * b_beta * exp_ge
    value = h_prev * tl.exp(gl + gel)[None, :]
    value += tl.dot((dout.to(tl.float32) * (qe * scale)[None, :]).to(tl.bfloat16), qg, out_dtype=tl.float32)
    value -= tl.dot((df * re[None, :]).to(tl.bfloat16), rg, out_dtype=tl.float32)
    return value


@triton.heuristics({
    'VARLEN': lambda args: args['cu'] is not None,
    'INITIAL': lambda args: args['dh0'] is not None,
    'FINAL': lambda args: args['dht'] is not None,
    'REG_STATE': lambda args: args['ws'] is None,
})
@triton.jit(do_not_specialize=['T', 'E'])
def chunk_triadic_kda_bwd_kernel_h(
    q2,
    k2,
    beta,
    gk,
    g2,
    Ai,
    Aqk,
    pq,
    pkp,
    pr,
    do,
    dx,
    dh,
    dht,
    dh0,
    ws,
    cu,
    offsets,
    scale,
    T,
    E,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E_CONST: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    VARLEN: tl.constexpr,
    INITIAL: tl.constexpr,
    FINAL: tl.constexpr,
    REG_STATE: tl.constexpr,
):
    iv, nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    n, h = nh // H, nh % H
    e_dim = E_CONST if REG_STATE else E.to(tl.int64)
    start, end, nc, first = sequence_bounds(n, T, cu, offsets, BT, VARLEN)

    t = tl.arange(0, BT).to(tl.int64)
    d = tl.arange(0, BK).to(tl.int64)
    vv = iv * BV + tl.arange(0, BV).to(tl.int64)
    ph = d[None, :] * V + vv[:, None]
    mh = (vv < V)[:, None] & (d < K)[None, :]

    if REG_STATE:
        hs = ()
        for e in tl.static_range(E_CONST):
            if FINAL:
                value = tl.load(dht + (nh * E_CONST + e) * K * V + ph, mh, other=0.0).to(tl.float32)
            else:
                value = tl.zeros((BV, BK), tl.float32)
            hs += (value,)
    else:
        for e in tl.range(e_dim):
            p_ws = (nh * e_dim + e) * K * V + ph
            value = tl.load(dht + p_ws, mh, other=0.0).to(tl.float32) if FINAL else tl.zeros((BV, BK), tl.float32)
            tl.store(ws + p_ws, value, mh)
        tl.debug_barrier()

    for c in tl.range(nc - 1, -1, -1):
        tt = start + c * BT + t
        mt = tt < end
        pk = (tt[:, None] * H + h) * K + d[None, :]
        mk = mt[:, None] & (d < K)[None, :]
        pv = (tt[None, :] * H + h) * V + vv[:, None]
        mv = (vv < V)[:, None] & mt[None, :]
        pe = (tt * H + h) * e_dim
        last = tl.minimum(start + (c + 1) * BT, end) - 1
        last_pe = (last * H + h) * e_dim
        chunk_head_base = ((first + c) * H + h) * e_dim

        kg_t = tl.trans(tl.load(pkp + pk, mk, other=0.0))
        dout = tl.load(do + pv, mv, other=0.0)
        pa = (tt[:, None] * H + h) * BT + t[None, :]

        # read pass: accumulate each slice's contribution and cache its state in bf16
        du = state_dot(dout, tl.load(Aqk + pa, mt[:, None], other=0.0)) * scale
        if REG_STATE:
            for e in tl.static_range(E_CONST):
                hp = hs[e].to(tl.bfloat16)
                tl.store(dh + (chunk_head_base + e) * K * V + ph, hp, mh)
                du += _bwd_read_slice(hp, e, pe, last_pe, mt, g2, k2, kg_t)
        else:
            for e in tl.range(e_dim):
                hp = tl.load(ws + (nh * e_dim + e) * K * V + ph, mh, other=0.0).to(tl.bfloat16)
                tl.store(dh + (chunk_head_base + e) * K * V + ph, hp, mh)
                du += _bwd_read_slice(hp, e, pe, last_pe, mt, g2, k2, kg_t)

        # intra-chunk inverse, then update each slice's state gradient
        df = state_dot(du, tl.load(Ai + pa, mt[:, None], other=0.0))
        tl.store(dx + pv, df, mv)

        rg = tl.load(pr + pk, mk, other=0.0)
        qg = tl.load(pq + pk, mk, other=0.0)
        gl = tl.load(gk + (last * H + h) * K + d, d < K, other=0.0)
        b_beta = tl.load(beta + tt * H + h, mt, other=0.0).to(tl.float32)

        if REG_STATE:
            hs_new = ()
            for e in tl.static_range(E_CONST):
                hs_new += (_bwd_update_slice(hs[e], e, pe, last_pe, mt, g2, q2, k2, b_beta, gl, dout, df, qg, rg, scale),)
            hs = hs_new
        else:
            for e in tl.range(e_dim):
                p_ws = (nh * e_dim + e) * K * V + ph
                h_cur = tl.load(ws + p_ws, mh, other=0.0)
                h_next = _bwd_update_slice(h_cur, e, pe, last_pe, mt, g2, q2, k2, b_beta, gl, dout, df, qg, rg, scale)
                tl.store(ws + p_ws, h_next, mh)
            tl.debug_barrier()

    if REG_STATE and INITIAL:
        for e in tl.static_range(E_CONST):
            tl.store(dh0 + (nh * E_CONST + e) * K * V + ph, hs[e], mh)


@triton.heuristics({
    'VARLEN': lambda args: args['cu'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_bwd_kernel_inter(
    q,
    k,
    q2,
    k2,
    beta,
    gk,
    g2,
    states,
    dh,
    do,
    dx,
    u,
    dq,
    dk,
    dr,
    dgk,
    dq2,
    dk2,
    dr2,
    dg2,
    cu,
    indices,
    scale,
    T,
    TOTAL_T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    VARLEN: tl.constexpr,
):
    ik, c, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    start, end, gc = chunk_position(c, bh // H, T, cu, indices, BT, VARLEN)
    h = bh % H

    t = tl.arange(0, BT).to(tl.int64)
    d = ik * BK + tl.arange(0, BK).to(tl.int64)
    tt = start + t
    mt, md = tt < end, d < K
    pk = (tt[:, None] * H + h) * K + d[None, :]
    mask = mt[:, None] & md[None, :]
    pe = (tt * H + h) * E
    last = tl.minimum(start + BT, end) - 1

    # fuse q and r contractions against the shared bf16 forward state cache
    g = tl.load(gk + pk, mask, other=0.0)
    decay_qr = tl.exp(g)
    acc_q = tl.zeros((BT, BK), tl.float32)
    acc_r = tl.zeros((BT, BK), tl.float32)
    b_beta = tl.load(beta + tt * H + h, mt, other=0.0).to(tl.float32)

    for e in tl.range(E):
        exp_ge = tl.exp(tl.load(g2 + pe + e, mt, other=0.0))
        contraction_q = tl.zeros((BT, BK), tl.float32)
        contraction_r = tl.zeros((BT, BK), tl.float32)
        for iv in range(tl.cdiv(V, BV)):
            vv = iv * BV + tl.arange(0, BV).to(tl.int64)
            pv = (tt[:, None] * H + h) * V + vv[None, :]
            mv = mt[:, None] & (vv < V)[None, :]
            ph = ((gc * H + h) * E + e) * K * V + d[:, None] * V + vv[None, :]
            state_t = tl.trans(tl.load(states + ph, md[:, None] & (vv < V)[None, :], other=0.0))
            contraction_q += tl.dot(tl.load(do + pv, mv, other=0.0).to(tl.bfloat16), state_t, out_dtype=tl.float32)
            contraction_r += tl.dot(tl.load(dx + pv, mv, other=0.0).to(tl.bfloat16), state_t, out_dtype=tl.float32)

        contraction_q = contraction_q * (exp_ge * scale)[:, None]
        contraction_r = -contraction_r * exp_ge[:, None]
        qe = tl.load(q2 + pe + e, mt, other=0.0)
        re = tl.load(k2 + pe + e, mt, other=0.0) * b_beta
        acc_q += contraction_q * decay_qr * qe[:, None]
        acc_r += contraction_r * decay_qr * re[:, None]
        qv = tl.load(q + pk, mask, other=0.0).to(tl.float32)
        kv = tl.load(k + pk, mask, other=0.0).to(tl.float32)
        pp = ((ik * TOTAL_T + tt) * H + h) * E + e
        tl.store(dq2 + pp, tl.sum(qv * decay_qr * contraction_q, 1), mt)
        tl.store(dr2 + pp, tl.sum(kv * decay_qr * contraction_r, 1), mt)

    tl.store(dq + pk, acc_q, mask)
    tl.store(dr + pk, acc_r, mask)
    tl.debug_barrier()

    # contract k against dh and accumulate the state-decay gradient
    g = tl.load(gk + pk, mask, other=0.0)
    gl = tl.load(gk + (last * H + h) * K + d, md, other=0.0)
    decay_k = tl.exp(tl.minimum(gl[None, :] - g, 0.0))
    kv = tl.load(k + pk, mask, other=0.0).to(tl.float32)
    acc_k = tl.zeros((BT, BK), tl.float32)
    last_grad = tl.zeros((BK,), tl.float32)

    for e in tl.range(E):
        ge = tl.load(g2 + pe + e, mt, other=0.0)
        gel = tl.load(g2 + (last * H + h) * E + e)
        contraction_k = tl.zeros((BT, BK), tl.float32)
        ss = tl.zeros((BK,), tl.float32)
        for iv in range(tl.cdiv(V, BV)):
            vv = iv * BV + tl.arange(0, BV).to(tl.int64)
            pv = (tt[:, None] * H + h) * V + vv[None, :]
            mv = mt[:, None] & (vv < V)[None, :]
            ph = ((gc * H + h) * E + e) * K * V + d[:, None] * V + vv[None, :]
            mh = md[:, None] & (vv < V)[None, :]
            dh_state = tl.load(dh + ph, mh, other=0.0)
            contraction_k += tl.dot(tl.load(u + pv, mv, other=0.0).to(tl.bfloat16), tl.trans(dh_state), out_dtype=tl.float32)
            ss += tl.sum(tl.load(states + ph, mh, other=0.0).to(tl.float32) * dh_state.to(tl.float32), 1)

        contraction_k = contraction_k * tl.exp(tl.minimum(gel - ge, 0.0))[:, None]
        ss = ss * tl.exp(gl + gel)
        ke = tl.load(k2 + pe + e, mt, other=0.0)
        acc_k += contraction_k * decay_k * ke[:, None]
        second_k = tl.sum(kv * decay_k * contraction_k, 1)
        pp = ((ik * TOTAL_T + tt) * H + h) * E + e
        tl.store(dk2 + pp, second_k, mt)
        qe = tl.load(q2 + pe + e, mt, other=0.0)
        vg2 = qe * tl.load(dq2 + pp, mt, other=0.0) + ke * b_beta * tl.load(dr2 + pp, mt, other=0.0) - ke * second_k
        last_grad += ss
        tl.store(dg2 + pp, vg2 + tl.where(tt == last, tl.sum(ss, 0) + tl.sum(ke * second_k, 0), 0.0), mt)

    tl.store(dk + pk, acc_k, mask)
    last_grad += tl.sum(kv * acc_k, 0)
    qv = tl.load(q + pk, mask, other=0.0).to(tl.float32)
    ag = -kv * acc_k + qv * tl.load(dq + pk, mask, other=0.0) + kv * tl.load(dr + pk, mask, other=0.0)
    tl.store(dgk + pk, ag + tl.where((tt == last)[:, None], last_grad[None, :], 0.0), mask)


@triton.heuristics({
    'VARLEN': lambda args: args['cu'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_bwd_kernel_matrices(
    do,
    dx,
    u,
    v,
    beta,
    dv,
    dbeta_v,
    dA,
    dM,
    cu,
    indices,
    scale,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    VARLEN: tl.constexpr,
):
    c, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    start, end, _ = chunk_position(c, bh // H, T, cu, indices, BT, VARLEN)
    h = bh % H
    t = tl.arange(0, BT).to(tl.int64)
    tt = start + t
    mt = tt < end

    da, dm = tl.zeros((BT, BT), tl.float32), tl.zeros((BT, BT), tl.float32)
    b_beta = tl.load(beta + tt * H + h, mt, other=0.0).to(tl.float32)
    dbeta_acc = tl.zeros((BT,), tl.float32)
    for iv in range(tl.cdiv(V, BV)):
        vv = iv * BV + tl.arange(0, BV).to(tl.int64)
        p = (tt[:, None] * H + h) * V + vv[None, :]
        mask = mt[:, None] & (vv < V)[None, :]
        uu_t = tl.trans(tl.load(u + p, mask, other=0.0)).to(tl.float32)
        dz = tl.load(dx + p, mask, other=0.0)
        da -= tl.dot(dz.to(tl.float32), uu_t, input_precision='ieee')
        dm += tl.dot(tl.load(do + p, mask, other=0.0).to(tl.float32), uu_t, input_precision='ieee') * scale
        tl.store(dv + p, dz * b_beta[:, None], mask)
        dbeta_acc += tl.sum(dz * tl.load(v + p, mask, other=0.0).to(tl.float32), 1)

    tl.store(dbeta_v + tt * H + h, dbeta_acc, mt)
    p = (tt[:, None] * H + h) * BT + t[None, :]
    tl.store(dA + p, tl.where(t[:, None] > t[None, :], da, 0.0), mt[:, None])
    tl.store(dM + p, tl.where(t[:, None] >= t[None, :], dm, 0.0), mt[:, None])


@triton.heuristics({
    'VARLEN': lambda args: args['cu'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_bwd_kernel_scores_e(
    q2,
    k2,
    beta,
    g2,
    Lk,
    Mk,
    dA,
    dM,
    dq,
    dk,
    dr,
    dg,
    outq,
    outk,
    dbeta_v,
    outg,
    cu,
    indices,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    VARLEN: tl.constexpr,
    NK: tl.constexpr,
    TOTAL_T: tl.constexpr,
):
    c, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    start, end, _ = chunk_position(c, bh // H, T, cu, indices, BT, VARLEN)
    h = bh % H
    t = tl.arange(0, BT).to(tl.int64)
    tt = start + t
    mask = tt < end
    pa = (tt[:, None] * H + h) * BT + t[None, :]
    base_l = tl.load(dA + pa, mask[:, None], other=0.0) * tl.load(Lk + pa, mask[:, None], other=0.0)
    base_m = tl.load(dM + pa, mask[:, None], other=0.0) * tl.load(Mk + pa, mask[:, None], other=0.0)
    b_beta = tl.load(beta + tt * H + h, mask, other=0.0).to(tl.float32)
    dbeta_acc = tl.load(dbeta_v + tt * H + h, mask, other=0.0).to(tl.float32)

    for e in tl.range(E):
        p = (tt * H + h) * E + e
        qi = tl.load(q2 + p, mask, other=0.0)
        ki = tl.load(k2 + p, mask, other=0.0)
        gi = tl.load(g2 + p, mask, other=0.0)
        ri = ki * b_beta

        first_g = tl.max(tl.where(t == 0, gi, float('-inf')), 0)
        last_g = tl.min(tl.where(mask, gi, float('inf')), 0)
        if first_g - last_g <= MAX_ANCHOR_SPAN:
            anchor = 0.5 * (first_g + last_g)
            decay = tl.exp(tl.where(mask, gi - anchor, 0.0))[:, None] * tl.exp(tl.where(mask, anchor - gi, 0.0))[None, :]
        else:
            decay = tl.exp(tl.minimum(gi[:, None] - gi[None, :], 0.0))

        dl, dm = base_l * decay, base_m * decay
        vq, vr = tl.sum(dm * ki[None, :], 1), tl.sum(dl * ki[None, :], 1)
        vk = tl.sum(dl * ri[:, None] + dm * qi[:, None], 0)
        diagonal = tl.sum(tl.where(t[:, None] == t[None, :], dm, 0.0), 1)

        inter_q, inter_k = tl.zeros((BT,), tl.float32), tl.zeros((BT,), tl.float32)
        inter_r, inter_g = tl.zeros((BT,), tl.float32), tl.zeros((BT,), tl.float32)
        for ik in tl.static_range(NK):
            offset = ik * TOTAL_T * H * E + p
            inter_q += tl.load(dq + offset, mask, other=0.0)
            inter_k += tl.load(dk + offset, mask, other=0.0)
            inter_r += tl.load(dr + offset, mask, other=0.0)
            inter_g += tl.load(dg + offset, mask, other=0.0)

        vg = qi * (vq - diagonal * ki) + ri * vr - ki * (vk - diagonal * qi) + inter_g
        erase = vr + inter_r
        tl.store(outq + p, vq + inter_q, mask)
        tl.store(outk + p, vk + inter_k + b_beta * erase, mask)
        dbeta_acc += ki * erase
        tl.store(outg + p, tl.cumsum(vg, reverse=True), mask)

    tl.store(dbeta_v + tt * H + h, dbeta_acc, mask)


@triton.jit
def _offdiag_scores_e(
    q2,
    k2,
    g2,
    b_beta_row,
    t_row,
    t_col,
    t_anchor,
    end,
    h,
    H: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    BC: tl.constexpr,
    TRANS: tl.constexpr,
):
    l2_sub = tl.zeros((BC, BC), tl.float32)
    m2_sub = tl.zeros((BC, BC), tl.float32)
    for ie in range(tl.cdiv(E, BE)):
        e_idx = ie * BE + tl.arange(0, BE).to(tl.int64)
        me_r = (t_row < end)[:, None] & (e_idx < E)[None, :]
        me_c = (t_col < end)[:, None] & (e_idx < E)[None, :]
        pe_r = (t_row[:, None] * H + h) * E + e_idx[None, :]
        pe_c = (t_col[:, None] * H + h) * E + e_idx[None, :]
        g2_n = tl.load(g2 + (t_anchor * H + h) * E + e_idx, e_idx < E, other=0.0)

        q2_r = tl.load(q2 + pe_r, me_r, other=0.0)
        r2_r = tl.load(k2 + pe_r, me_r, other=0.0) * b_beta_row[:, None]
        fg_r = tl.where(me_r, tl.exp(tl.minimum(tl.load(g2 + pe_r, me_r, other=0.0) - g2_n[None, :], 0.0)), 0.0)
        k2_c = tl.load(k2 + pe_c, me_c, other=0.0)
        fg_c = tl.where(me_c, tl.exp(tl.minimum(g2_n[None, :] - tl.load(g2 + pe_c, me_c, other=0.0), 0.0)), 0.0)
        k2_scaled = k2_c * fg_c
        if TRANS:
            l2_sub = tl.dot(k2_scaled, tl.trans(r2_r * fg_r), l2_sub, input_precision='ieee')
            m2_sub = tl.dot(k2_scaled, tl.trans(q2_r * fg_r), m2_sub, input_precision='ieee')
        else:
            k2_gt = tl.trans(k2_scaled)
            l2_sub = tl.dot(r2_r * fg_r, k2_gt, l2_sub, input_precision='ieee')
            m2_sub = tl.dot(q2_r * fg_r, k2_gt, m2_sub, input_precision='ieee')
    return l2_sub, m2_sub


@triton.heuristics({
    'VARLEN': lambda args: args['cu'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_bwd_kernel_scores_k(
    q,
    k,
    g,
    q2,
    k2,
    beta,
    g2,
    dA,
    dM,
    dq,
    dk,
    dr,
    dg,
    outq,
    outk,
    outg,
    cu,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BD: tl.constexpr,
    BE: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    VARLEN: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64)
    c, block = block // (BT // BC), block % (BT // BC)
    bh, kd = tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    start, end, _ = chunk_position(c, bh // H, T, cu, indices, BT, VARLEN)
    if start + block * BC >= end:
        return
    h = bh % H

    oi = tl.arange(0, BC).to(tl.int64)
    i = block * BC + oi
    d = kd * BD + tl.arange(0, BD).to(tl.int64)
    ti = start + i
    md = d < K
    mi = (ti < end)[:, None] & md[None, :]
    pi = (ti[:, None] * H + h) * K + d[None, :]
    qi = tl.load(q + pi, mi, other=0.0)
    ki = tl.load(k + pi, mi, other=0.0)
    ri = ki.to(tl.float32)
    gi = tl.load(g + pi, mi, other=0.0)
    vq, vr, vk = tl.zeros((BC, BD), tl.float32), tl.zeros((BC, BD), tl.float32), tl.zeros((BC, BD), tl.float32)
    b_beta_i = tl.load(beta + ti * H + h, ti < end, other=0.0).to(tl.float32)

    # off-diagonal sub-blocks before this block: dq and dr
    if block > 0:
        t_anchor = start + block * BC
        gn = tl.load(g + (t_anchor * H + h) * K + d, md, other=0.0)
        factor_q = tl.exp(tl.minimum(gi - gn[None, :], 0.0))
        for jb in range(0, block):
            oj = jb * BC + oi
            tjb = start + oj
            mjb = (tjb < end)[:, None] & md[None, :]
            pjb = (tjb[:, None] * H + h) * K + d[None, :]
            kjb = tl.load(k + pjb, mjb, other=0.0) * tl.exp(tl.minimum(gn[None, :] - tl.load(g + pjb, mjb, other=0.0), 0.0))
            l2_sub, m2_sub = _offdiag_scores_e(q2, k2, g2, b_beta_i, ti, tjb, t_anchor, end, h, H, E, BE, BC, False)
            pa = (ti[:, None] * H + h) * BT + oj[None, :]
            ma = (ti < end)[:, None] & (tjb < end)[None, :]
            vr += state_dot(tl.load(dA + pa, ma, other=0.0) * l2_sub, kjb) * factor_q
            vq += state_dot(tl.load(dM + pa, ma, other=0.0) * m2_sub, kjb) * factor_q

    # off-diagonal sub-blocks after this block: dk only
    if start + (block + 1) * BC < end and block < BT // BC - 1:
        t_anchor = start + (block + 1) * BC - 1
        gn = tl.load(g + (t_anchor * H + h) * K + d, md, other=0.0)
        factor_k = tl.exp(tl.minimum(gn[None, :] - gi, 0.0))
        for jb in range(block + 1, BT // BC):
            oj = jb * BC + oi
            tjb = start + oj
            mjb = (tjb < end)[:, None] & md[None, :]
            pjb = (tjb[:, None] * H + h) * K + d[None, :]
            qjb = tl.load(q + pjb, mjb, other=0.0)
            rjb = tl.load(k + pjb, mjb, other=0.0).to(tl.float32)
            fg = tl.where((tjb < end)[:, None], tl.exp(tl.minimum(tl.load(g + pjb, mjb, other=0.0) - gn[None, :], 0.0)), 0.0)
            b_beta_j = tl.load(beta + tjb * H + h, tjb < end, other=0.0).to(tl.float32)
            l2_sub_t, m2_sub_t = _offdiag_scores_e(q2, k2, g2, b_beta_j, tjb, ti, t_anchor, end, h, H, E, BE, BC, True)
            pa = (tjb[None, :] * H + h) * BT + i[:, None]
            ma = (tjb < end)[None, :] & (ti < end)[:, None]
            dl = tl.load(dA + pa, ma, other=0.0) * l2_sub_t
            dm = tl.load(dM + pa, ma, other=0.0) * m2_sub_t
            vk += (state_dot(dl, rjb * fg) + state_dot(dm, qjb * fg)) * factor_k

    # diagonal [BC, BC] sub-block
    l2_d = tl.zeros((BC, BC), tl.float32)
    m2_d = tl.zeros((BC, BC), tl.float32)
    for ie in range(tl.cdiv(E, BE)):
        e_idx = ie * BE + tl.arange(0, BE).to(tl.int64)
        me_d = e_idx < E
        me_i = (ti < end)[:, None] & me_d[None, :]
        pe_i = (ti[:, None] * H + h) * E + e_idx[None, :]
        q2_i = tl.load(q2 + pe_i, me_i, other=0.0)
        k2_i = tl.load(k2 + pe_i, me_i, other=0.0)
        g2_i = tl.load(g2 + pe_i, me_i, other=0.0)
        l2_d, m2_d = diag_score_tile(k2_i * b_beta_i[:, None], q2_i, k2_i, g2_i, me_i, me_d, l2_d, m2_d, BC)

    pa = (ti[:, None] * H + h) * BT + i[None, :]
    ma = (ti < end)[:, None] & (ti < end)[None, :] & (oi[:, None] >= oi[None, :])
    dl_d = tl.load(dA + pa, ma, other=0.0) * tl.where(ma, l2_d, 0.0)
    dm_d = tl.load(dM + pa, ma, other=0.0) * tl.where(ma, m2_d, 0.0)

    first_g = tl.where(md, tl.max(tl.where(oi[:, None] == 0, gi, float('-inf')), 0), 0.0)
    last_g = tl.where(md, tl.min(tl.where(mi, gi, float('inf')), 0), 0.0)
    if tl.max(first_g - last_g, 0) <= MAX_ANCHOR_SPAN:
        anchor = 0.5 * (first_g + last_g)
        gq = tl.where(mi, tl.exp(gi - anchor[None, :]), 0.0)
        gk_inv = tl.where(mi, tl.exp(anchor[None, :] - gi), 0.0)
        k_scaled = ri * gk_inv
        vq += tl.dot(dm_d, k_scaled, input_precision='ieee') * gq
        vr += tl.dot(dl_d, k_scaled, input_precision='ieee') * gq
        vk += (
            tl.dot(tl.trans(dl_d), ri * gq, input_precision='ieee')
            + tl.dot(tl.trans(dm_d), qi.to(tl.float32) * gq, input_precision='ieee')
        ) * gk_inv
    else:
        for col in range(BC):
            m_col = oi == col
            g_col = tl.sum(tl.where(m_col[:, None], gi, 0.0), 0)
            r_col = tl.sum(tl.where(m_col[:, None], ri, 0.0), 0)
            dl_col = tl.sum(tl.where(m_col[None, :], dl_d, 0.0), 1)
            dm_col = tl.sum(tl.where(m_col[None, :], dm_d, 0.0), 1)
            decay = tl.exp(tl.minimum(gi - g_col[None, :], 0.0))
            kg = r_col[None, :] * decay
            vq += dm_col[:, None] * kg
            vr += dl_col[:, None] * kg
            vk_col = tl.sum((dl_col[:, None] * ri + dm_col[:, None] * qi.to(tl.float32)) * decay, 0)
            vk += tl.where(m_col[:, None], vk_col[None, :], 0.0)

    vg = qi * vq + ri * vr - ki * vk
    erase = tl.load(dr + pi, mi, other=0.0) + vr
    tl.store(outq + pi, tl.load(dq + pi, mi, other=0.0) + vq, mi)
    tl.store(outk + pi, tl.load(dk + pi, mi, other=0.0) + vk + erase, mi)
    tl.store(outg + pi, tl.load(dg + pi, mi, other=0.0) + vg, mi)


def chunk_triadic_kda_bwd(v, beta, q, k, q2, k2, gk, g2, Ai, Aqk, states, u, Lk, Mk, h0, cu, indices, do, dht, scale, keys):
    B, T, H, K = k.shape
    E, V = k2.shape[-1], v.shape[-1]
    N = len(cu) - 1 if cu is not None else B
    nc = len(indices) if cu is not None else triton.cdiv(T, CHUNK_SIZE)
    offsets = prepare_chunk_offsets(cu, CHUNK_SIZE) if cu is not None else None
    inter_block = 32
    nk = triton.cdiv(K, inter_block)

    if states is None:
        keys = [torch.empty_like(k, dtype=torch.bfloat16) for _ in range(3)]
        Akk = k.new_empty(B, T, H, CHUNK_SIZE, dtype=torch.float32)
        Lk, Mk = fwd_intra(
            q=q,
            k=k,
            q2=q2,
            k2=k2,
            beta=beta,
            gk=gk,
            g2=g2,
            cu=cu,
            indices=indices,
            Akk=Akk,
            Aqk=Aqk,
            keys=keys,
        )
        _, _, states, u = fwd_h(
            v=v,
            beta=beta,
            q2=q2,
            k2=k2,
            gk=gk,
            g2=g2,
            Ai=Ai,
            Aqk=Aqk,
            keys=keys,
            h0=h0,
            cu=cu,
            indices=indices,
            scale=scale,
            save=True,
            output=False,
            final=False,
        )

    pq, pkp, pr = keys
    dx, dh = torch.empty_like(u), torch.empty_like(states)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    use_reg = E <= MAX_REG_E
    ws = None if use_reg else (dh0 if dh0 is not None else u.new_empty(N, H, E, K, V, dtype=torch.float32))
    BV = get_state_block_size(N, H, V, E)

    chunk_triadic_kda_bwd_kernel_h[(triton.cdiv(V, BV), N * H)](
        q2=q2,
        k2=k2,
        beta=beta,
        gk=gk,
        g2=g2,
        Ai=Ai,
        Aqk=Aqk,
        pq=pq,
        pkp=pkp,
        pr=pr,
        do=do,
        dx=dx,
        dh=dh,
        dht=dht,
        dh0=dh0,
        ws=ws,
        cu=cu,
        offsets=offsets,
        scale=scale,
        T=T,
        E=E,
        H=H,
        K=K,
        V=V,
        E_CONST=E if use_reg else 0,
        BT=CHUNK_SIZE,
        BK=max(16, triton.next_power_of_2(K)),
        BV=BV,
        num_warps=8,
        num_stages=1,
    )

    inter_k = [torch.empty_like(k, dtype=torch.float32) for _ in range(4)]
    inter_e = [k.new_empty(nk, B, T, H, E, dtype=torch.float32) for _ in range(4)]
    chunk_triadic_kda_bwd_kernel_inter[(nk, nc, B * H)](
        q=q,
        k=k,
        q2=q2,
        k2=k2,
        beta=beta,
        gk=gk,
        g2=g2,
        states=states,
        dh=dh,
        do=do,
        dx=dx,
        u=u,
        dq=inter_k[0],
        dk=inter_k[1],
        dr=inter_k[2],
        dgk=inter_k[3],
        dq2=inter_e[0],
        dk2=inter_e[1],
        dr2=inter_e[2],
        dg2=inter_e[3],
        cu=cu,
        indices=indices,
        scale=scale,
        T=T,
        TOTAL_T=B * T,
        H=H,
        K=K,
        V=V,
        E=E,
        BT=CHUNK_SIZE,
        BK=inter_block,
        BV=32 if V >= 32 else 16,
        num_warps=4,
        num_stages=2,
    )

    dA, dM = [torch.empty_like(Aqk, dtype=torch.float32) for _ in range(2)]
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta, dtype=torch.float32)
    dq2, dk2, dg2 = torch.empty_like(k2), torch.empty_like(k2), torch.empty_like(k2)
    chunk_triadic_kda_bwd_kernel_matrices[(nc, B * H)](
        do=do,
        dx=dx,
        u=u,
        v=v,
        beta=beta,
        dv=dv,
        dbeta_v=dbeta,
        dA=dA,
        dM=dM,
        cu=cu,
        indices=indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=CHUNK_SIZE,
        BV=32,
        num_warps=8,
        num_stages=1,
    )

    dq, dk, dgk = torch.empty_like(q), torch.empty_like(k), torch.empty_like(gk)
    chunk_triadic_kda_bwd_kernel_scores_e[(nc, B * H)](
        q2=q2,
        k2=k2,
        beta=beta,
        g2=g2,
        Lk=Lk,
        Mk=Mk,
        dA=dA,
        dM=dM,
        dq=inter_e[0],
        dk=inter_e[1],
        dr=inter_e[2],
        dg=inter_e[3],
        outq=dq2,
        outk=dk2,
        dbeta_v=dbeta,
        outg=dg2,
        cu=cu,
        indices=indices,
        T=T,
        H=H,
        E=E,
        BT=CHUNK_SIZE,
        NK=nk,
        TOTAL_T=B * T,
        num_warps=8,
    )
    chunk_triadic_kda_bwd_kernel_scores_k[(nc * 4, B * H, nk)](
        q=q,
        k=k,
        g=gk,
        q2=q2,
        k2=k2,
        beta=beta,
        g2=g2,
        dA=dA,
        dM=dM,
        dq=inter_k[0],
        dk=inter_k[1],
        dr=inter_k[2],
        dg=inter_k[3],
        outq=dq,
        outk=dk,
        outg=dgk,
        cu=cu,
        indices=indices,
        T=T,
        H=H,
        K=K,
        E=E,
        BD=32,
        BE=min(64, max(16, triton.next_power_of_2(E))),
        BT=CHUNK_SIZE,
        BC=16,
        num_warps=4,
        num_stages=1,
    )

    dgk = chunk_local_cumsum(dgk, chunk_size=CHUNK_SIZE, reverse=True, cu_seqlens=cu, chunk_indices=indices)
    return dq, dk, dv, dk2, dq2, dgk, dg2, dbeta, dh0
