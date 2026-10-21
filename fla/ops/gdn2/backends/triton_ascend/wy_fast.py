# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN-2 WY recompute for triton-ascend.

The shared kernel dots ``A`` with activation tiles directly. On NPU ``A`` is
kept in fp32 while ``k`` / ``v`` stay in fp16 or bf16, and Ascend ``tl.dot``
rejects that mix. This copy promotes the right-hand side to fp32 and reloads
``A`` before every dot, because Ascend ``tl.dot`` can clobber its left operand.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2


@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_gdn2_kernel_npu(
    q,
    k,
    qg,
    kg,
    v,
    b,
    w_gate,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_A = A + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_A[None, :]

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_u = u + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_wg = w_gate + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)
        b_wg = tl.load(p_wg, mask=m_v, other=0.0).to(tl.float32)
        b_vb = b_v * b_wg
        b_A = tl.load(p_A, mask=m_A, other=0.0).to(tl.float32)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_kk = m_t[:, None] & (o_k[None, :] < K)
        p_w = w + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_b = b + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kk, other=0.0).to(tl.float32)
        b_b = tl.load(p_b, mask=m_kk, other=0.0).to(tl.float32)
        b_kb = b_k * b_b

        p_gk = gk + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_gk = tl.load(p_gk, mask=m_kk, other=0.0).to(tl.float32)
        b_kb *= exp2(b_gk)

        if STORE_QG:
            p_q = q + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
            p_qg = qg + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_kk, other=0.0)
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), mask=m_kk)

        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k,
                mask=m_k, other=0.,
            ).to(tl.float32)
            b_kg = b_k * tl.where(
                (i_t * BT + tl.arange(0, BT) < T)[:, None],
                exp2(b_gn[None, :] - b_gk),
                0,
            )
            p_kg = kg + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), mask=m_kk)

        b_A = tl.load(p_A, mask=m_A, other=0.0).to(tl.float32)
        b_w = tl.dot(b_A, b_kb.to(tl.float32), allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_kk)


def recompute_w_u_fwd_gdn2_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """NPU WY recompute. ``A @ x`` and the returned factors stay in fp32.

    The inter-chunk state is allocated from ``k``'s dtype. Writing ``w`` / ``u`` /
    ``kg`` back to bf16 rounds that state on every chunk and pushes a long
    sequence past the bf16 reference tolerance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty(B, T, H, K, device=k.device, dtype=torch.float32)
    u = torch.empty(B, T, H, V, device=v.device, dtype=torch.float32)
    qg = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32) if q is not None else None
    kg = torch.empty(B, T, H, K, device=k.device, dtype=torch.float32) if gk is not None else None
    recompute_w_u_fwd_gdn2_kernel_npu[(NT, B * H)](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        v=v,
        b=b,
        w_gate=w_gate,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u, qg, kg
