# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# WY-representation recompute for GDN-2.
#
# Given the solved chunk inverse ``A = (I + tril(T,-1))^{-1}``, this kernel
# produces the two within-chunk auxiliaries consumed by the inter-chunk state
# recurrence:
#
#   u = A @ (w * v)            # write-side, V-axis
#   w = A @ (b * exp(gk) * k)  # erase-side, K-axis
#
# It optionally emits ``qg = q * exp(gk)`` and the tail-decayed key
# ``kg = k * exp(gn - gk)`` used by the inter-chunk update.
#
# Difference vs the KDA WY recompute: KDA uses a single scalar ``beta`` for both
# the u and w blocks. GDN-2 uses the channel-wise ``w_gate`` on the V axis for u
# and the channel-wise ``b`` on the K axis for w. These two gates live on
# different axes and cannot be reduced to a shared scalar; the kernel loads
# them per BV / BK block respectively.

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_gdn2_kernel(
    q,
    k,
    qg,
    kg,
    v,
    b,           # channel-wise erase gate, [B, T, H, K]
    w_gate,      # channel-wise write gate, [B, T, H, V]
    w,           # output: A @ (b * exp(gk) * k), [B, T, H, K]
    u,           # output: A @ (w_gate * v), [B, T, H, V]
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1),
        (i_t * BT, 0), (BT, BT), (1, 0),
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0),
        )
        p_wg = tl.make_block_ptr(
            w_gate + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_wg = tl.load(p_wg, boundary_check=(0, 1))
        b_vb = (b_v * b_wg).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_b = tl.make_block_ptr(
            b + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        b_kb = b_k * b_b

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb *= exp2(b_gk)

        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K, (T, K), (H * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K, (T, K), (H * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))

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
            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K, (T, K), (H * K, 1),
                (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd_gdn2(
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
    """Produce GDN-2 WY auxiliaries from the solved chunk inverse ``A``.

    Returns ``(w, u, qg, kg)`` with shapes matching the inputs. ``qg`` is
    produced only when ``q`` is provided; ``kg`` only when ``gk`` is provided.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    qg = torch.empty_like(q) if q is not None else None
    kg = torch.empty_like(k) if gk is not None else None
    recompute_w_u_fwd_gdn2_kernel[(NT, B * H)](
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
