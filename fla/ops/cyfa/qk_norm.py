# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# Portions adapted from CyclicFlowAttention, Copyright (c) 2026 Yixiao Chen.
# https://github.com/Chyxx/CyclicFlowAttention

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import autotune_cache_kwargs


@fla_cache_autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=num_warps)
        for bt in [32, 64, 128]
        for num_warps in [4, 8]
    ],
    key=["D", "STORE_Q", "STORE_RSTD"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["N"])
def _qk_rmsnorm_fwd_kernel(
    q,
    k,
    q_weight,
    k_weight,
    q_out,
    k_out,
    q_rstd,
    k_rstd,
    q_eps,
    k_eps,
    N,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    STORE_Q: tl.constexpr,
    STORE_RSTD: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    o_n = i_n * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_n = o_n < N
    m_d = o_d < D
    mask = m_n[:, None] & m_d[None, :]
    offsets = o_n[:, None] * D + o_d[None, :]

    b_q = tl.load(q + offsets, mask=mask, other=0.0).to(tl.float32)
    b_q = tl.where(mask, b_q, 0.0)
    b_q_rstd = tl.rsqrt(tl.sum(b_q * b_q, axis=1) / D + q_eps)
    b_q_weight = tl.load(q_weight + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_Q:
        tl.store(q_out + offsets, b_q * b_q_rstd[:, None] * b_q_weight[None, :], mask=mask)
    if STORE_RSTD:
        tl.store(q_rstd + o_n, b_q_rstd, mask=m_n)

    b_k = tl.load(k + offsets, mask=mask, other=0.0).to(tl.float32)
    b_k = tl.where(mask, b_k, 0.0)
    b_k_rstd = tl.rsqrt(tl.sum(b_k * b_k, axis=1) / D + k_eps)
    b_k_weight = tl.load(k_weight + o_d, mask=m_d, other=0.0).to(tl.float32)
    tl.store(k_out + offsets, b_k * b_k_rstd[:, None] * b_k_weight[None, :], mask=mask)
    if STORE_RSTD:
        tl.store(k_rstd + o_n, b_k_rstd, mask=m_n)


def qk_rmsnorm_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    *,
    store_q: bool = True,
    store_rstd: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if q.shape != k.shape or q.shape[-1] != q_weight.numel() or q.shape[-1] != k_weight.numel():
        raise ValueError("Fused Q/K RMSNorm requires matching contiguous Q/K tensors and weights.")
    q = q.contiguous()
    k = k.contiguous()
    q_out = torch.empty_like(q) if store_q else q
    k_out = torch.empty_like(k)
    n_rows = q.numel() // q.shape[-1]
    q_rstd = torch.empty(n_rows, device=q.device, dtype=torch.float32) if store_rstd else None
    k_rstd = torch.empty(n_rows, device=q.device, dtype=torch.float32) if store_rstd else None

    def grid(meta):
        return (triton.cdiv(n_rows, meta["BT"]),)

    _qk_rmsnorm_fwd_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        q_out,
        k_out,
        q_rstd,
        k_rstd,
        float(q_eps),
        float(k_eps),
        N=n_rows,
        D=q.shape[-1],
        BD=triton.next_power_of_2(q.shape[-1]),
        STORE_Q=bool(store_q),
        STORE_RSTD=bool(store_rstd),
    )
    return q_out, k_out, q_rstd, k_rstd
