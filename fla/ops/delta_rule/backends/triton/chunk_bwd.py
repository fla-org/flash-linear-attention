# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused delta-rule WY + dq/dk backward experiment.

This is an op-local producer-consumer fusion for the dense ungated delta-rule
backward. It keeps the WY consumer of ``dw = -(dv @ h)`` in the same program
that produces ``dw`` so the wide ``dw`` tensor is never materialized to HBM.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import safe_dot


@triton.jit(do_not_specialize=["T"])
def _chunk_delta_rule_wy_dqkw_fused_kernel(
    q,
    k,
    v,
    v_new,
    beta,
    A,
    h,
    do,
    dh,
    dv,
    dq,
    dk,
    dv2,
    dbeta,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    NT = tl.cdiv(T, BT)
    i_tg = (i_b * NT + i_t).to(tl.int64)
    bos = (i_b * T).to(tl.int64)
    t0 = i_t * BT

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    beta += bos * H + i_h
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K * V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K * V
    dv += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv2 += (bos * H + i_h) * V
    dbeta += bos * H + i_h

    o_t = t0 + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (t0,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)

    # Existing prepare_wy_repr_bwd consumes A transposed.
    p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (0, t0), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)

    # K-independent work: local dS plus the V-side WY backward contribution.
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (t0, i_v * BV), (BT, BV), (1, 0))
        p_v_new = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (t0, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (t0, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (t0, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (H * V, 1), (t0, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dv = tl.load(p_dv, boundary_check=(0, 1))

        b_ds += tl.dot(b_do, tl.trans(b_v_new))
        b_dA += tl.dot(b_dv, tl.trans(b_v))

        b_dvb = tl.dot(b_A, b_dv)
        b_dv2 = b_dvb * b_beta[:, None]
        b_dbeta += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))

    m_lower = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_ds = tl.where(m_lower, b_ds, 0).to(q.dtype.element_ty)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (t0, i_k * BK), (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (t0, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (t0, i_v * BV), (BT, BV), (1, 0))
            p_v_new = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (t0, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (t0, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))

            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))

            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
            b_dw += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))

        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale

        b_dw = -b_dw.to(b_A.dtype)
        b_dA += tl.dot(b_dw, tl.trans(b_k.to(b_A.dtype)))

        b_dk_beta = tl.dot(b_A, b_dw)
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        b_dk += b_dk_beta * b_beta[:, None]

        p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (t0, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (t0, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    m_strict = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_dA = tl.where(m_strict, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_strict, -b_dA, 0)

    # Final transformed-dA terms for dk/dbeta. This stays in the fused launch,
    # avoiding the separate prepare_wy_repr_bwd + dk.add_ HBM round trip.
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (t0, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (t0, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

        b_dk_beta = tl.dot(b_dA.to(b_k.dtype), b_k)
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        b_dk += safe_dot(tl.trans(b_dA.to(b_k.dtype)), b_k_beta, allow_tf32=False)
        b_dk += b_dk_beta * b_beta[:, None]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    p_dbeta = tl.make_block_ptr(dbeta, (T,), (H,), (t0,), (BT,), (0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))


def chunk_delta_rule_wy_dqkw_fused_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size
    BK = 64
    BV = 64
    if scale is None:
        scale = K ** -0.5

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv2 = torch.empty_like(v)
    dbeta = torch.empty_like(beta)

    h_flat = h.reshape(-1, h.shape[-2], h.shape[-1])
    dh_flat = dh.reshape(-1, dh.shape[-2], dh.shape[-1])
    grid = (triton.cdiv(T, BT), B * H)
    _chunk_delta_rule_wy_dqkw_fused_kernel[grid](
        q,
        k,
        v,
        v_new,
        beta,
        A,
        h_flat,
        do,
        dh_flat,
        dv,
        dq,
        dk,
        dv2,
        dbeta,
        scale,
        T,
        H,
        K,
        V,
        BT,
        BK,
        BV,
        num_warps=4,
        num_stages=3,
    )
    return dq, dk, dv2, dbeta
