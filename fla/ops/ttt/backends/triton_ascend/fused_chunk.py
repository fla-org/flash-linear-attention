# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend split of the fused-chunk TTT linear backward dh kernel.

``fused_chunk_ttt_linear_bwd_kernel_dh`` holds ~244KiB of live tiles at
chunk_size 64 (BT/BK/BV pinned by semantics), overflowing the 192KiB Ascend
Unified Buffer. The dh-independent cluster (b_A, b_Ae, b_ds and their
dh-free consumers in dk/de) is split into a chunk-parallel pre-pass kernel;
the remaining sequential kernel (the backward dh recurrence plus the
group-norm backward chain) then fits the budget. The math and the chunk
granularity are unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.utils import ascend_compile_kwargs


@triton.jit(do_not_specialize=['T'])
def fused_chunk_ttt_linear_bwd_kernel_dh_part1(
    q,
    k,
    do,
    v2,
    eta,
    dvnl,
    dkdsp,
    de_loc,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_pid = tl.program_id(0).to(tl.int64)
    NT = tl.cdiv(T, BT)
    i_t = i_pid % NT
    i_nh = i_pid // NT
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    m_q = (o_k[:, None] < K) & m_t[None, :]
    m_k = m_t[:, None] & (o_k[None, :] < K)
    m_v = m_t[:, None] & (o_v[None, :] < V)
    m_A = o_i[:, None] >= o_i[None, :]
    m_A_t = o_i[:, None] <= o_i[None, :]
    p_q = q+(bos*H+i_h)*K + o_k[:, None] + o_t[None, :] * (H*K)
    p_k = k+(bos*H+i_h)*K + o_t[:, None] * (H*K) + o_k[None, :]
    p_do = do+(bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
    p_v2 = v2+(bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
    p_e = eta+(bos*H+i_h) + o_t * H
    p_dvnl = dvnl+(bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
    p_dkdsp = dkdsp+(bos*H+i_h)*K + o_t[:, None] * (H*K) + o_k[None, :]
    p_de_loc = de_loc+(bos*H+i_h) + o_t * H
    b_q = tl.load(p_q, mask=m_q, other=0.0)
    b_k = tl.load(p_k, mask=m_k, other=0.0)
    b_e = tl.load(p_e, mask=m_t, other=0.0)
    b_do = tl.load(p_do, mask=m_v, other=0.0)
    b_v2 = tl.load(p_v2, mask=m_v, other=0.0)

    b_A = tl.dot(b_k, b_q)
    b_A = - tl.where(m_A_t, b_A * scale * b_e[None, :], 0).to(do.dtype.element_ty)
    b_Ae = - tl.where(m_A_t, b_e[None, :], 0).to(do.dtype.element_ty)
    b_dvnl = tl.dot(b_A.to(b_do.dtype), b_do) + tl.dot(b_Ae.to(b_do.dtype), b_do)
    tl.store(p_dvnl, b_dvnl.to(p_dvnl.dtype.element_ty), mask=m_v)

    b_ds = tl.dot(b_do, tl.trans(b_v2))
    b_ds = tl.where(m_A, b_ds, 0)
    b_ds = b_ds.to(b_k.dtype)
    b_qs = (b_q * scale).to(b_q.dtype)
    b_dkdsp = tl.dot(tl.trans(b_ds), tl.trans(b_qs) * b_e[:, None])
    tl.store(p_dkdsp, b_dkdsp.to(p_dkdsp.dtype.element_ty), mask=m_k)
    b_de_loc = tl.sum(tl.dot(b_ds, b_k) * tl.trans(b_qs).to(b_k.dtype), axis=1) + tl.sum(b_ds, axis=1)
    tl.store(p_de_loc, b_de_loc.to(p_de_loc.dtype.element_ty), mask=m_t)


@triton.jit(do_not_specialize=['T'])
def fused_chunk_ttt_linear_bwd_kernel_dh_part2(
    q,
    k,
    v,
    v2,
    x,
    y,
    r,
    w,
    b,
    eta,
    h,
    dht,
    dhbt,
    dh0,
    dhb0,
    do,
    dvnl,
    dkdsp,
    de_loc,
    dk,
    dv,
    de,
    dw,
    db,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT_B: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    bos, _ = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_dhb = tl.zeros([BV], dtype=tl.float32)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_h = (o_k[:, None] < K) & (o_v[None, :] < V)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_dh += tl.load(p_dht, mask=m_h, other=0.0)
    if USE_FINAL_STATE_GRADIENT_B:
        p_dhbt = dhbt + i_nh * V + o_v
        b_dhb += tl.load(p_dhbt, mask=o_v < V, other=0.0)

    # [BV]
    o_i = tl.arange(0, BT)
    v_i = tl.arange(0, BV)
    b_w = tl.load(w + i_h * V + v_i, mask=v_i < V, other=0.)
    b_b = tl.load(b + i_h * V + v_i, mask=v_i < V, other=0.)
    b_dw = tl.zeros([BV], dtype=b_w.dtype)
    b_db = tl.zeros([BV], dtype=b_b.dtype)
    p_dw = dw + i_nh * V + o_v
    p_db = db + i_nh * V + o_v

    for i_t in range(NT - 1, -1, -1):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_q = (o_k[:, None] < K) & m_t[None, :]
        m_k = m_t[:, None] & (o_k[None, :] < K)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        m_hv = (o_v[:, None] < V) & (o_k[None, :] < K)
        p_h = h+((boh+i_t)*H+i_h)*K*V + o_v[:, None] + o_k[None, :] * V
        p_q = q+(bos*H+i_h)*K + o_k[:, None] + o_t[None, :] * (H*K)
        p_k = k+(bos*H+i_h)*K + o_t[:, None] * (H*K) + o_k[None, :]
        p_v = v+(bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
        p_v2 = v2+(bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
        p_x = x+(bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
        p_y = y+(bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
        p_r = r+bos*H+i_h + o_t[:, None] * H
        p_dv = dv+(bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
        p_dk = dk+(bos*H+i_h)*K + o_t[:, None] * (H*K) + o_k[None, :]
        p_do = do+(bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
        p_de = de+(bos*H+i_h) + o_t * H
        p_dvnl = dvnl+(bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
        p_dkdsp = dkdsp+(bos*H+i_h)*K + o_t[:, None] * (H*K) + o_k[None, :]
        p_de_loc = de_loc+(bos*H+i_h) + o_t * H
        p_e_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        b_q = tl.load(p_q, mask=m_q, other=0.0)
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_do = tl.load(p_do, mask=m_v, other=0.0)
        b_e_last = tl.load(p_e_last)
        # dh-independent terms (b_A, b_Ae, b_ds and their dh-free consumers)
        # are precomputed by fused_chunk_ttt_linear_bwd_kernel_dh_part1.
        b_dv_new = tl.load(p_dvnl, mask=m_v, other=0.0).to(tl.float32)
        b_dv_new -= tl.dot(b_e_last * b_k, b_dh.to(b_k.dtype))
        b_dv_new -= b_e_last * b_dhb.to(b_k.dtype)[None, :]

        b_v2 = tl.load(p_v2, mask=m_v, other=0.0).to(b_k.dtype)
        b_x = tl.load(p_x, mask=m_v, other=0.0).to(b_k.dtype)
        b_y = tl.load(p_y, mask=m_v, other=0.0).to(b_k.dtype)
        b_rstd = tl.load(p_r, mask=m_t[:, None], other=0.0).to(tl.float32)
        b_dy = b_rstd * (b_dv_new * V - tl.sum(b_dv_new, axis=1, keep_dims=True) -
                         b_x * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_dx = -b_rstd * (b_dv_new * tl.sum(b_x * b_y, axis=1, keep_dims=True) +
                          b_y * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_drstd = tl.sum(b_dv_new.to(b_rstd.dtype) * b_v2.to(b_rstd.dtype) / b_rstd, axis=1, keep_dims=True)

        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_w = b_w.to(b_k.dtype)
        b_b = b_b.to(b_k.dtype)
        b_dv = -b_w * b_dy.to(b_k.dtype)
        b_dk = b_w * b_dy.to(b_k.dtype)
        b_dw += tl.sum(2 * b_w * b_x * b_dy.to(b_k.dtype) +
                       (b_b - b_v.to(b_k.dtype) + b_k) * b_dy.to(b_k.dtype), axis=0).to(b_dw.dtype)
        b_db += tl.sum(b_w * b_dy.to(b_k.dtype), axis=0).to(b_db.dtype)
        b_dx = b_dx.to(b_k.dtype) + b_w * b_w * b_dy.to(b_k.dtype)

        b_h = tl.load(p_h, mask=m_hv, other=0.0)
        b_q = (b_q * scale).to(b_q.dtype)
        b_dkh = b_rstd * (V * b_dx - tl.sum(b_dx, axis=1, keep_dims=True) -
                          b_x * tl.sum(b_x * b_dx, axis=1, keep_dims=True)) / V
        b_dkh -= b_rstd * b_rstd * b_drstd * b_x / V
        b_dkh = tl.where((v_i < V)[None, :] * (o_i < T-i_t*BT)[:, None], b_dkh, 0.)
        b_dk += tl.dot(b_dkh, b_h.to(b_dkh.dtype)).to(b_k.dtype)

        b_dkdsp = tl.load(p_dkdsp, mask=m_k, other=0.0)
        b_de_loc = tl.load(p_de_loc, mask=m_t, other=0.0)
        i_last = (BT-1) if (i_t*BT+BT) <= T else (T % BT-1)
        mask = (o_i == i_last)
        b_dk -= b_e_last * tl.dot(b_v2, tl.trans(b_dh).to(b_v2.dtype))
        b_dk -= b_dkdsp
        b_de = tl.where(mask, tl.sum(- b_dh * tl.trans(tl.dot(tl.trans(b_v2), b_k))), 0.0)
        b_de -= tl.where(mask, tl.sum(b_dhb * tl.sum(b_v2, axis=0)), 0.0)
        b_de -= b_de_loc
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype)) + tl.dot(tl.trans(b_k).to(b_dkh.dtype), b_dkh)
        b_dhb += tl.sum(b_do + b_dkh, axis=0)
        b_dh = tl.where((v_i < V)[None, :], b_dh, 0.)
        b_dhb = tl.where((v_i < V), b_dhb, 0.)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_v)
        tl.store(p_de, b_de.to(p_de.dtype.element_ty), mask=m_t)
    tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), mask=o_v < V)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=o_v < V)

    if USE_INITIAL_STATE:
        p_dh0 = dh0+i_nh*K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=m_h)
    if USE_INITIAL_STATE_B:
        p_dhb0 = dhb0+i_nh*V + o_v
        tl.store(p_dhb0, b_dhb.to(p_dhb0.dtype.element_ty), mask=o_v < V)


def fused_chunk_ttt_linear_bwd_dh_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v2: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    h: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dhbt: torch.Tensor,
    BT: int = 16,
    initial_state: torch.Tensor = None,
    initial_state_bias: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    assert cu_seqlens is None, "bwd of varlen is not implemented yet."
    B, T, H, K, V = *k.shape, v.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    N = B
    NT = triton.cdiv(T, BT)
    BK, BV = max(triton.next_power_of_2(K), 16), max(triton.next_power_of_2(V), 16)
    assert max(BK, BV) <= 128, "current kernel does not support head dimension larger than 128."

    dh0 = torch.empty_like(initial_state, dtype=torch.float32) if initial_state is not None else None
    dhb0 = torch.empty_like(initial_state_bias, dtype=torch.float32) if initial_state_bias is not None else None
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    de = torch.empty_like(eta)
    dw = w.new_empty(B, H, V)
    db = b.new_empty(B, H, V)
    # dh-independent pre-pass outputs (see the module docstring).
    dvnl = torch.empty(B, T, H, V, dtype=do.dtype, device=do.device)
    dkdsp = torch.empty(B, T, H, K, dtype=do.dtype, device=do.device)
    de_loc = torch.empty(B, T, H, dtype=do.dtype, device=do.device)

    fused_chunk_ttt_linear_bwd_kernel_dh_part1[(NT * N * H,)](
        q=q,
        k=k,
        do=do,
        v2=v2,
        eta=eta,
        dvnl=dvnl,
        dkdsp=dkdsp,
        de_loc=de_loc,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        **ascend_compile_kwargs(),
    )
    fused_chunk_ttt_linear_bwd_kernel_dh_part2[(N * H,)](
        q=q,
        k=k,
        v=v,
        v2=v2,
        x=x,
        y=y,
        r=r,
        w=w,
        b=b,
        eta=eta,
        h=h,
        dht=dht,
        dhbt=dhbt,
        dh0=dh0,
        dhb0=dhb0,
        do=do,
        dvnl=dvnl,
        dkdsp=dkdsp,
        de_loc=de_loc,
        dk=dk,
        dv=dv,
        de=de,
        dw=dw,
        db=db,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        USE_INITIAL_STATE_B=initial_state_bias is not None,
        USE_FINAL_STATE_GRADIENT=dht is not None,
        USE_FINAL_STATE_GRADIENT_B=dhbt is not None,
        **ascend_compile_kwargs(),
    )
    dw = dw.sum(dim=0)
    db = db.sum(dim=0)
    return dk, dv, de, dw, db, dh0, dhb0


__all__ = ['fused_chunk_ttt_linear_bwd_dh_npu']
