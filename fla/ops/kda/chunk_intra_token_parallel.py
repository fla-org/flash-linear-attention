# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Token-parallel implementation of KDA intra chunk kernel

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp, exp2
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]  # Let autotune choose freely
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'B'])
def chunk_kda_fwd_kernel_intra_token_parallel(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk_diag,
    scale,
    cu_seqlens,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Each block processes one token (i) for BH heads
    i_tg = tl.program_id(0)  # global token index
    i_hg = tl.program_id(1)  # head_group index

    i_h_start = i_hg * BH

    if IS_VARLEN:
        # Binary search to find which sequence this token belongs to
        # i_tg is the global token index
        # Range [0, B) where B is num_sequences passed from python

        left = 0
        right = B
        i_n = 0

        # Unrolled binary search (max B=2^32)
        # We can limit iterations based on expected max batch size if needed
        # 20 iterations covers B=1M, usually enough
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                end_val = tl.load(cu_seqlens + mid + 1).to(tl.int32)
                if i_tg < end_val:
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        i_t = i_tg - bos
        T = eos - bos  # Current sequence length

        # Safety check
        if i_t >= T or i_tg >= eos:
            return

    else:
        i_b = i_tg // T
        i_t = i_tg % T
        bos = i_b * T

        if i_t >= T:
            return

    i_chunk = i_t // BT  # which BT=64 chunk
    i_subchunk = (i_t % BT) // BC  # which BC=16 sub-chunk within the BT chunk

    subchunk_start = i_chunk * BT + i_subchunk * BC
    subchunk_end = tl.minimum(subchunk_start + BC, T)

    o_h = tl.arange(0, BH)
    m_h = (i_h_start + o_h) < H

    # Marginalize over entire K dimension at once
    BK: tl.constexpr = triton.next_power_of_2(K)
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    # Load q[i_t, h:h+BH, :] - shape [BH, K]
    # For varlen, we use global offset: bos + i_t = i_tg
    p_q = tl.make_block_ptr(q + (bos + i_t) * H * K, (H, K), (K, 1), (i_h_start, 0), (BH, BK), (0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)  # [BH, BK]

    # Load g[i_t, h:h+BH, :]
    p_g = tl.make_block_ptr(g + (bos + i_t) * H * K, (H, K), (K, 1), (i_h_start, 0), (BH, BK), (0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)  # [BH, BK]

    # Load k[i_t, h:h+BH, :] and beta[i_t, h:h+BH]
    p_k = tl.make_block_ptr(k + (bos + i_t) * H * K, (H, K), (K, 1), (i_h_start, 0), (BH, BK), (0, 1))
    b_k_self = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)  # [BH, BK]

    p_beta = beta + (bos + i_t) * H + i_h_start + o_h
    b_beta = tl.load(p_beta, mask=m_h, other=0).to(tl.float32)  # [BH]
    b_k_self = b_k_self * b_beta[:, None]  # [BH, K]

    for j in range(subchunk_start, tl.minimum(i_t + 1, subchunk_end)):

        # Load k[j, h:h+BH, :] with pointer arithmetic
        p_kj = tl.make_block_ptr(k + (bos + j) * H * K, (H, K), (K, 1), (i_h_start, 0), (BH, BK), (0, 1))
        b_kj = tl.load(p_kj, boundary_check=(0, 1)).to(tl.float32)  # [BH, BK]

        # Load g[j, h:h+BH, :]
        p_gj = tl.make_block_ptr(g + (bos + j) * H * K, (H, K), (K, 1), (i_h_start, 0), (BH, BK), (0, 1))
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)  # [BH, BK]

        # Compute gated key for all BH heads: [BH, BK]
        if USE_EXP2:
            b_kgj = b_kj * exp2(b_g - b_gj)
        else:
            b_kgj = b_kj * exp(b_g - b_gj)

        # Apply mask for valid K dimension
        b_kgj = tl.where(m_k[None, :], b_kgj, 0.0)

        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale  # [BH]
        # Akk: only accumulate if j < i_t
        b_Akk = tl.sum(b_k_self * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)  # [BH]

        # Store Aqk with [B, T, H, BT] layout
        j_pos = j % BT
        offs_h = i_h_start + o_h
        offs_aqk = (bos + i_t) * H * BT + offs_h * BT + j_pos
        tl.store(Aqk + offs_aqk, b_Aqk.to(Aqk.dtype.element_ty), mask=m_h)

        # Store Akk_diag with [B, T, H, BC] layout (only diagonal blocks)
        j_pos_diag = j - subchunk_start  # position within sub-chunk [0, BC)
        offs_akk = (bos + i_t) * H * BC + offs_h * BC + j_pos_diag
        tl.store(Akk_diag + offs_akk, b_Akk.to(Akk_diag.dtype.element_ty), mask=m_h)


def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk_diag: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
    use_exp2: bool = False,
) -> None:
    """
    Token-parallel implementation: each token gets its own thread block.
    Supports both fixed-length and variable-length sequences.
    Reduces wasted computation on padding.

    Writes directly to Aqk and Akk_diag tensors (in-place).

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        gk: [B, T, H, K] cumsum of gates
        beta: [B, T, H]
        Aqk: [B, T, H, BT] output tensor to write to
        Akk_diag: [B, T, H, BC] output tensor for diagonal blocks (fp32)
        scale: attention scale
        chunk_size: BT (default 64)
        sub_chunk_size: BC (default 16)
        use_exp2: use exp2 vs exp
    """
    B, T, H, K = q.shape
    BT = chunk_size
    BC = sub_chunk_size

    # Grid: (total_tokens, H/BH) - each token gets its own block
    if cu_seqlens is not None:
        total_tokens = q.shape[1]
        # Use num_sequences as B for binary search
        B_kernel = len(cu_seqlens) - 1
    else:
        total_tokens = B * T
        B_kernel = B

    def grid(meta):
        BH = meta['BH']
        return (total_tokens, triton.cdiv(H, BH))

    chunk_kda_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk_diag=Akk_diag,
        scale=scale,
        cu_seqlens=cu_seqlens,
        B=B_kernel,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        USE_EXP2=use_exp2,
    )
