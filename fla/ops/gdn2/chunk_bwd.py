# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

r"""
Backward pipeline for GDN-2 chunkwise training.

This file owns the GDN-2-specific WY-aware backward kernel and the end-to-end
backward orchestration. The inter-chunk state-recurrence backward
(``chunk_gated_delta_rule_bwd_dhu``) and the dAv backward
(``chunk_kda_bwd_dAv``) are shared with KDA and imported as-is.

The crux of the GDN-2 backward is the WY-aware vector-Jacobian product through
``A = (I + tril(T,-1))^{-1}``. For a scalar write strength ``beta`` the
contribution of ``u`` to ``dA`` factors as ``dU @ (beta * V)^T = beta * (dU @
V^T)``, so KDA can apply ``beta`` as a row-scalar post-multiply. GDN-2 replaces
``beta`` with two channel-wise gates ``b`` (key axis) and ``w_gate`` (value
axis) that live on different axes; no scalar post-scale can recover them. The
kernel therefore bakes the gates directly into the dA accumulation:

    dA += dU @ (w_gate * V)^T
    dA += dW @ (b * exp(gk) * K)^T

and produces the partial gradients db (channel-wise, K-dim) and dw (channel-
wise, V-dim) at the same time.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.gdn2.chunk_intra import chunk_gdn2_bwd_intra
from fla.ops.gdn2.wy_fast import recompute_w_u_fwd_gdn2
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs

NUM_WARPS_WY = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in NUM_WARPS_WY
        for num_stages in [2, 3, 4]
        if not (IS_NVIDIA_HOPPER and BK == 32 and num_warps == 4)
    ],
    key=['BT', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gdn2_bwd_kernel_wy_dqkg_fused(
    q,
    k,
    v,
    v_new,
    g,
    b,
    w_gate,
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dv2,
    dg,
    db,
    dw,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    w_gate += (bos * H + i_h) * V
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K * V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    db += (bos * H + i_h) * K
    dw += (bos * H + i_h) * V
    dA += (bos * H + i_h) * BT

    p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_b = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_b = tl.load(p_b, boundary_check=(0, 1))

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw_flow = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            p_v_new = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
            b_dw_flow += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))
            tl.debug_barrier()

            if i_k == 0:
                p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_wg = tl.make_block_ptr(w_gate, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_dw_gate = tl.make_block_ptr(dw, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_wg = tl.load(p_wg, boundary_check=(0, 1))
                # dA gets (w_gate * v) on the value side - the GDN-2 channel-wise twist.
                b_dA += tl.dot(b_dv, tl.trans(b_v * b_wg))

                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_wg
                b_dw_gate = b_dvb * b_v

                tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dw_gate, b_dw_gate.to(p_dw_gate.dtype.element_ty), boundary_check=(0, 1))

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_b
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)

        b_kg = b_k * b_gk_exp

        b_dw_flow = -b_dw_flow.to(b_A.dtype)
        # dA gets (b * exp(gk) * k) on the key side - the GDN-2 channel-wise twist.
        b_dA += tl.dot(b_dw_flow, tl.trans((b_kg * b_b).to(b_A.dtype)))

        b_dkgb = tl.dot(b_A, b_dw_flow)
        p_db = tl.make_block_ptr(db, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_db_partial = b_dkgb * b_kg
        tl.store(p_db, b_db_partial.to(p_db.dtype.element_ty), boundary_check=(0, 1))

        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_b
        b_dk = b_dk + b_dkgb * b_gb

        p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)
    p_dA = tl.make_block_ptr(dA, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


def chunk_gdn2_bwd_wy_dqkg_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    state_v_first: bool = False,
):
    """Fused WY backward producing dq, dk, dv, db (K-dim), dw (V-dim), dg, dA."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float32)
    db = torch.empty_like(b, dtype=torch.float32)
    dw = torch.empty_like(w_gate, dtype=torch.float32)
    dA = torch.empty_like(A, dtype=torch.float32)

    grid = (NT, B * H)
    chunk_gdn2_bwd_kernel_wy_dqkg_fused[grid](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        b=b,
        w_gate=w_gate,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=dv,
        dv2=dv2,
        dg=dg,
        db=db,
        dw=dw,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    dv = dv2
    return dq, dk, dv, db, dw, dg, dA


def chunk_gdn2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    g: torch.Tensor | None = None,
    g_org: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    state_v_first: bool = False,
    w_wy: torch.Tensor | None = None,
    u_wy: torch.Tensor | None = None,
    qg: torch.Tensor | None = None,
    kg: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
    h: torch.Tensor | None = None,
    disable_recompute: bool = False,
):
    """End-to-end GDN-2 backward.

    Returns (dq, dk, dv, db, dw, dg, dh0, dA_log, dt_bias_grad). ``db`` has
    shape [B, T, H, K] (channel-wise erase gate); ``dw`` has shape [B, T, H, V]
    (channel-wise write gate).
    """
    if not disable_recompute:
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
            )
        w_wy, u_wy, qg, kg = recompute_w_u_fwd_gdn2(
            k=k,
            v=v,
            b=b,
            w_gate=w_gate,
            A=Akk,
            q=q,
            gk=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w_wy,
            u=u_wy,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
        )

    dAqk, dv = chunk_kda_bwd_dAv(
        q=q,
        k=k,
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w_wy,
        gk=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
    )

    dq, dk, dv, db, dw, dg, dAkk = chunk_gdn2_bwd_wy_dqkg_fused(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        b=b,
        w_gate=w_gate,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )

    dq, dk, db, dg = chunk_gdn2_bwd_intra(
        q=q,
        k=k,
        g=g,
        b=b,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )

    dA_log, dt_bias_grad = None, None
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if use_gate_in_kernel:
        dg, dA_log, dt_bias_grad = kda_gate_bwd(
            g=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=dg,
            lower_bound=lower_bound,
        )

    return dq, dk, dv, db, dw, dg, dh0, dA_log, dt_bias_grad
