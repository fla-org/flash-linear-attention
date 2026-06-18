# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'N'])
def chunk_precond_kda_fwd_kernel_intra_token_parallel(
    q,
    k,           # Original k (for Akk row side)
    k_precond,   # Preconditioned k (for column side of both Aqk and Akk)
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Asymmetric version for precond_kda:
    - Aqk = q @ k_precond^T (column uses k_precond)
    - Akk = k @ k_precond^T (row uses original k, column uses k_precond)
    """
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N

        # Unrolled binary search (max B=2^32)
        # We can limit iterations based on expected max batch size if needed
        # 20 iterations covers B=1M, usually enough
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    q += bos * H*K
    k += bos * H*K
    k_precond += bos * H*K
    g += bos * H*K
    Aqk += bos * H*BT
    Akk += bos * H*BC
    beta += bos * H

    BK: tl.constexpr = triton.next_power_of_2(K)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = (i_hg * BH + o_h) < H
    m_k = o_k < K

    p_q = tl.make_block_ptr(q + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta + i_t * H, (H,), (1,), (i_hg * BH,), (BH,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)  # Original k for row
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_k = b_k * tl.load(p_beta, boundary_check=(0,)).to(tl.float32)[:, None]

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        # Load k_precond at position j (column side) - ASYMMETRIC: use k_precond here
        p_kpj = tl.make_block_ptr(k_precond + j * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        p_gj = tl.make_block_ptr(g + j * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))

        b_kpj = tl.load(p_kpj, boundary_check=(0, 1)).to(tl.float32)  # k_precond for column
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        # k_precond gated at position j
        b_kpgj = b_kpj * exp2(b_g - b_gj)
        b_kpgj = tl.where(m_k[None, :], b_kpgj, 0.0)

        # Aqk: q (row i) @ k_precond^T (col j)
        b_Aqk = tl.sum(b_q * b_kpgj, axis=1) * scale
        # Akk: k (row i, beta-scaled) @ k_precond^T (col j) - ASYMMETRIC
        b_Akk = tl.sum(b_k * b_kpgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

        tl.store(Aqk + i_t * H*BT + (i_hg * BH + o_h) * BT + j % BT, b_Aqk.to(Aqk.dtype.element_ty), mask=m_h)
        tl.store(Akk + i_t * H*BC + (i_hg * BH + o_h) * BC + j - i_ts, b_Akk.to(Akk.dtype.element_ty), mask=m_h)


def chunk_precond_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    k_precond: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> None:
    """
    Token-parallel implementation for preconditioned KDA.
    Supports both fixed-length and variable-length sequences.
    Reduces wasted computation on padding.
    Asymmetric: uses k for Akk row side, k_precond for column side.

    Writes directly to Aqk and Akk tensors (in-place).

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K] - original k (for Akk row side)
        k_precond: [B, T, H, K] - preconditioned k (for column side)
        gk: [B, T, H, K] cumsum of gates
        beta: [B, T, H]
        Aqk: [B, T, H, BT] output tensor
        Akk: [B, T, H, BC] output tensor for diagonal blocks (fp32)
        scale: attention scale
    """
    B, T, H, K = q.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    def grid(meta): return (B * T, triton.cdiv(H, meta['BH']))
    chunk_precond_kda_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        k_precond=k_precond,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )
    return Aqk, Akk
