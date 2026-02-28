# Copyright (c) 2023-2025, Chenguang Li
# NPU (Ascend) optimized implementation of chunk_gla

"""
NPU-optimized implementation of chunk_gla.

This module provides NPU-specific triton kernels and functions for GLA chunk operations.
The kernels are optimized for Huawei Ascend NPU architecture with adjusted block sizes
and tuning parameters.
"""

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cumsum import chunk_local_cumsum
from fla.ops.utils.op import exp, exp2
from fla.utils import autotune_cache_kwargs, input_guard

# Import the intra-chunk kernel from the GPU version since it can work on NPU too
from fla.ops.gla.chunk import chunk_gla_fwd_intra_gk

# NPU-specific block size configurations
# NPU typically has different memory hierarchy, so we adjust block sizes
BK_LIST_NPU = [16, 32, 64]  # Smaller block sizes for NPU
BV_LIST_NPU = [32, 64, 128]  # Adjusted for NPU


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST_NPU
        for BV in BV_LIST_NPU
        for num_warps in [2, 4]  # NPU-specific warp configuration
        for num_stages in [2, 3]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_kernel_o_npu(
    q,
    v,
    g,
    h,
    o,
    A,
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
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    NPU-optimized kernel for computing output o.
    
    This kernel computes the output tensor o from queries, values, gates, and attention matrix.
    """
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK] - Compute q*g with proper dtype handling for NPU
        b_g_f32 = b_g.to(tl.float32)
        if USE_EXP2:
            b_qg_f32 = b_q.to(tl.float32) * exp2(b_g_f32)
        else:
            b_qg_f32 = b_q.to(tl.float32) * exp(b_g_f32)
        b_qg = b_qg_f32.to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV] - Direct accumulation without condition
        b_o += tl.dot(b_qg.to(tl.float32), b_h.to(tl.float32))
    b_o *= scale
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT] - Load and mask A
    b_A = tl.load(p_A, boundary_check=(0, 1))
    # Use tl.where to properly replace upper triangle with zeros
    # This is critical because A matrix has uninitialized values in upper triangle
    # and 0 * uninitialized (Inf/NaN) = NaN, whereas tl.where properly replaces values
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A.to(tl.float32), b_v.to(tl.float32))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_o_gk_npu(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
):
    """NPU-optimized version of chunk_gla_fwd_o_gk."""
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Please ensure zeros, since vllm will use padding v
    o = torch.zeros_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_gla_fwd_kernel_o_npu[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
    )
    return o


def chunk_gla_fwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NPU-optimized forward pass for chunk_gla."""
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, chunk_size, cu_seqlens=cu_seqlens)

    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=None,
        gk=g_cumsum,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        # Use states_in_fp32=True for numerical precision in backward pass
        # This avoids recomputing h in backward (which fails on NPU)
        states_in_fp32=True,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    # the intra A is kept in fp32
    # Use the GPU version of chunk_gla_fwd_intra_gk which includes both
    # inter-block and intra-block (diagonal) computations
    A = chunk_gla_fwd_intra_gk(
        q=q,
        k=k,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    # NPU fix: Zero out the upper triangle of A to avoid uninitialized memory issues
    # The GPU version uses torch.empty() which leaves upper triangle with garbage
    # This garbage can cause NaN in backward pass even with tl.where masking
    BT = chunk_size
    T_dim = A.shape[1]
    
    if cu_seqlens is not None:
        # Varlen mode: compute local position within each sequence
        # For each position t, find which sequence it belongs to and compute local position
        cu_seqlens_cpu = cu_seqlens.cpu()
        num_seqs = len(cu_seqlens_cpu) - 1
        
        # Create a tensor of local positions within chunks
        local_pos = torch.zeros(T_dim, dtype=torch.long, device=A.device)
        for seq_idx in range(num_seqs):
            seq_start = cu_seqlens_cpu[seq_idx].item()
            seq_end = cu_seqlens_cpu[seq_idx + 1].item()
            seq_len = seq_end - seq_start
            if seq_len > 0:
                # Local positions within this sequence, mod BT for chunk-local position
                seq_positions = torch.arange(seq_len, device=A.device) % BT
                local_pos[seq_start:seq_end] = seq_positions
        
        j_idx = torch.arange(BT, device=A.device)
        causal_mask = j_idx[None, :] <= local_pos[:, None]  # [T, BT]
    else:
        # Regular batched mode
        t_idx = torch.arange(T_dim, device=A.device)
        j_idx = torch.arange(BT, device=A.device)
        # Create causal mask: j <= (t % BT) for valid positions
        causal_mask = j_idx[None, :] <= (t_idx % BT)[:, None]  # [T, BT]
    
    A = A * causal_mask[None, :, None, :].to(A.dtype)  # [B, T, H, BT]
    o = chunk_gla_fwd_o_gk_npu(
        q=q,
        v=v,
        g=g_cumsum,
        A=A,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return g_cumsum, A, h, ht, o


# NPU-specific implementation of chunk_gla_bwd_dqk_intra
# The triton kernel has issues on NPU with cross-sub-block computation
def chunk_gla_bwd_dqk_intra_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,  # g_cumsum
    dA: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    """
    NPU-specific implementation of chunk_gla_bwd_dqk_intra.
    
    Uses pure PyTorch to avoid triton kernel issues on NPU.
    The triton kernel has problems with cross-sub-block computation
    (the `if i_i > 0` branch produces incorrect results on NPU).
    
    Supports both regular batched input and variable-length sequences (cu_seqlens).
    """
    B, T, H, K = q.shape
    BT = chunk_size
    
    # Initialize output tensors
    dq = torch.zeros(B, T, H, K, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, T, H, K, dtype=torch.float32, device=q.device)
    
    if cu_seqlens is not None:
        # Variable length mode: process each sequence independently
        # cu_seqlens has shape [N+1] where N is number of sequences
        # cu_seqlens[i] is the start position of sequence i
        num_seqs = cu_seqlens.shape[0] - 1
        cu_seqlens_cpu = cu_seqlens.cpu()
        
        for seq_idx in range(num_seqs):
            seq_start = cu_seqlens_cpu[seq_idx].item()
            seq_end = cu_seqlens_cpu[seq_idx + 1].item()
            seq_len = seq_end - seq_start
            
            if seq_len == 0:
                continue
            
            # Extract sequence data (B=1 for varlen, data is concatenated along T)
            q_seq = q[:, seq_start:seq_end, :, :]  # [1, seq_len, H, K]
            k_seq = k[:, seq_start:seq_end, :, :]
            g_seq = g[:, seq_start:seq_end, :, :]
            dA_seq = dA[:, seq_start:seq_end, :, :]
            
            # Process this sequence
            dq_seq, dk_seq = _chunk_gla_bwd_dqk_intra_single(
                q_seq, k_seq, g_seq, dA_seq, BT
            )
            
            # Store results
            dq[:, seq_start:seq_end] = dq_seq
            dk[:, seq_start:seq_end] = dk_seq
    else:
        # Regular batched mode
        dq, dk = _chunk_gla_bwd_dqk_intra_single(q, k, g, dA, BT)
    
    return dq, dk


def _chunk_gla_bwd_dqk_intra_single(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    dA: torch.Tensor,
    BT: int,
):
    """
    Process a single batch/sequence for intra-chunk backward.
    
    Args:
        q, k, g: [B, T, H, K]
        dA: [B, T, H, BT]
        BT: chunk size
    
    Returns:
        dq, dk: [B, T, H, K]
    """
    B, T, H, K = q.shape
    
    # Compute number of chunks
    NT = (T + BT - 1) // BT
    T_padded = NT * BT
    
    # Pad if necessary (pad T dimension, which is dim 1 for [B, T, H, K])
    if T < T_padded:
        pad_size = T_padded - T
        # For [B, T, H, K], pad format is (K_left, K_right, H_left, H_right, T_left, T_right)
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad_size))
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_size))
        g = torch.nn.functional.pad(g, (0, 0, 0, 0, 0, pad_size))
        # dA has shape [B, T, H, BT], so pad format is (BT_left, BT_right, H_left, H_right, T_left, T_right)
        dA = torch.nn.functional.pad(dA, (0, 0, 0, 0, 0, pad_size))
    
    # Initialize output tensors
    dq = torch.zeros(B, T_padded, H, K, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, T_padded, H, K, dtype=torch.float32, device=q.device)
    
    # Create causal mask once
    causal_mask = torch.tril(torch.ones(BT, BT, device=q.device, dtype=torch.bool))
    
    # Process each chunk
    for n in range(NT):
        t_start = n * BT
        t_end = t_start + BT
        
        # Get chunk data and permute to [B, H, BT, K] for easier manipulation
        g_perm = g[:, t_start:t_end, :, :].permute(0, 2, 1, 3).float()  # [B, H, BT, K]
        k_perm = k[:, t_start:t_end, :, :].permute(0, 2, 1, 3).float()  # [B, H, BT, K]
        q_perm = q[:, t_start:t_end, :, :].permute(0, 2, 1, 3).float()  # [B, H, BT, K]
        dA_perm = dA[:, t_start:t_end, :, :].permute(0, 2, 1, 3).float()  # [B, H, BT, BT]
        
        # exp_factors[b, h, i, j, k] = exp(g[b, h, i, k] - g[b, h, j, k])
        g_i = g_perm[:, :, :, None, :]  # [B, H, BT, 1, K]
        g_j = g_perm[:, :, None, :, :]  # [B, H, 1, BT, K]
        # Clamp to avoid numerical overflow (exp of large values -> inf -> nan)
        exp_arg = g_i - g_j
        exp_arg = torch.clamp(exp_arg, max=20.0)  # exp(20) ≈ 4.85e8, safe for float32
        exp_factors = torch.exp(exp_arg)  # [B, H, BT, BT, K]
        
        # Apply causal mask: only j <= i should contribute
        exp_factors = exp_factors * causal_mask[None, None, :, :, None].float()
        
        # dq[i, k] = sum_j dA[i, j] * k[j, k] * exp(g[i, k] - g[j, k])
        k_exp = k_perm[:, :, None, :, :] * exp_factors  # [B, H, BT, BT, K]
        dq_chunk = torch.einsum('bhij,bhijk->bhik', dA_perm, k_exp)
        
        # dk[j, k] = sum_{i>=j} dA[i, j] * q[i, k] * exp(g[i, k] - g[j, k])
        q_exp = q_perm[:, :, :, None, :] * exp_factors  # [B, H, BT, BT, K]
        dk_chunk = torch.einsum('bhij,bhijk->bhjk', dA_perm, q_exp)
        
        # Store results (permute back to [B, BT, H, K])
        dq[:, t_start:t_end] = dq_chunk.permute(0, 2, 1, 3)
        dk[:, t_start:t_end] = dk_chunk.permute(0, 2, 1, 3)
    
    # Return only the valid part (trim padding)
    return dq[:, :T], dk[:, :T]


# For backward pass, we use the GPU implementation with some NPU-specific fixes
def chunk_gla_bwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    """
    NPU backward pass for chunk_gla.
    Uses the GPU backward helper functions.
    """
    # Import backward helper functions from GPU implementation
    from fla.ops.gla.chunk import (
        chunk_gla_bwd_dA,
        chunk_gla_bwd_dqkg,
        chunk_gla_bwd_dv,
    )
    
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, chunk_size, cu_seqlens=cu_seqlens)

    if h is None:
        h, _ = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=g_cumsum,
            gv=None,
            h0=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            states_in_fp32=True,
        )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=g_cumsum,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=True,
    )

    dv = chunk_gla_bwd_dv(
        k=k,
        g=g_cumsum,
        A=A,
        do=do,
        dh=dh,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    # dq dk in fp32
    dA = chunk_gla_bwd_dA(
        v=v,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    # NPU fix: Zero out the upper triangle of dA to avoid uninitialized memory issues
    # The GPU version uses torch.empty() which leaves upper triangle with garbage
    # and relies on tl.where in the intra kernel to mask it, but NPU may not handle this correctly
    BT = chunk_size
    T_dim = dA.shape[1]
    
    if cu_seqlens is not None:
        # Varlen mode: compute local position within each sequence
        cu_seqlens_cpu = cu_seqlens.cpu()
        num_seqs = len(cu_seqlens_cpu) - 1
        
        local_pos = torch.zeros(T_dim, dtype=torch.long, device=dA.device)
        for seq_idx in range(num_seqs):
            seq_start = cu_seqlens_cpu[seq_idx].item()
            seq_end = cu_seqlens_cpu[seq_idx + 1].item()
            seq_len = seq_end - seq_start
            if seq_len > 0:
                seq_positions = torch.arange(seq_len, device=dA.device) % BT
                local_pos[seq_start:seq_end] = seq_positions
        
        j_idx = torch.arange(BT, device=dA.device)
        causal_mask_dA = j_idx[None, :] <= local_pos[:, None]  # [T, BT]
    else:
        t_idx = torch.arange(T_dim, device=dA.device)
        j_idx = torch.arange(BT, device=dA.device)
        # dA[t, j] should be 0 when j > (t % BT), i.e., only lower triangle is valid
        causal_mask_dA = j_idx[None, :] <= (t_idx % BT)[:, None]  # [T, BT]
    
    dA = dA * causal_mask_dA[None, :, None, :].to(dA.dtype)  # [B, T, H, BT]
    
    # Use NPU-specific implementation to avoid triton kernel issues
    dq, dk = chunk_gla_bwd_dqk_intra_npu(
        q=q,
        k=k,
        g=g_cumsum,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dq, dk, dg = chunk_gla_bwd_dqkg(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g_cumsum,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return dq, dk, dv, dg, dh0


class ChunkGLAFunctionNPU(torch.autograd.Function):
    """NPU-optimized autograd function for chunk_gla."""

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    ):
        chunk_size = min(64, max(16, triton.next_power_of_2(q.shape[1])))

        g_cumsum, A, h, ht, o = chunk_gla_fwd_npu(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=None,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        # recompute g_cumsum in bwd pass
        if g.dtype != torch.float:
            g_cumsum = None
        else:
            g = None
        # Save h (computed in fp32) to use in backward
        # This avoids recomputing h in backward which can fail on NPU
        ctx.save_for_backward(q, k, v, g, g_cumsum, initial_state, A, h)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o, ht

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        q, k, v, g, g_cumsum, initial_state, A, h = ctx.saved_tensors
        chunk_size, scale, cu_seqlens = ctx.chunk_size, ctx.scale, ctx.cu_seqlens
        # Use saved h (already in fp32 from forward) for numerical precision
        dq, dk, dv, dg, dh0 = chunk_gla_bwd_npu(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=g_cumsum,
            scale=scale,
            h=h,  # Use saved fp32 h
            A=A,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        return dq.to(q), dk.to(k), dv.to(v), dg, None, dh0, None, None


@torch.compiler.disable
def chunk_gla_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: int | None = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    NPU-optimized chunk-based Gated Linear Attention (GLA).
    
    This is the NPU backend implementation of chunk_gla, optimized for
    Huawei Ascend NPUs. The interface is identical to the GPU version.
    
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, T, H, K]`.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."
    o, final_state = ChunkGLAFunctionNPU.apply(q, k, v, g, scale, initial_state, output_final_state, cu_seqlens)
    return o, final_state
