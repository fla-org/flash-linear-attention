# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp, gather
from fla.utils import IS_AMD, IS_GATHER_SUPPORTED, USE_CUDA_GRAPH, autotune_cache_kwargs

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [2, 4, 8, 16, 32]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'BT'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    a,
    b,
    gi,
    ge,
    qg,
    kg,
    ag,
    bg,
    Aqk,
    Aqb,
    Aab,
    Aak,
    cu_seqlens,
    chunk_indices,
    scale: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr,
):
    i_t, i_b, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + tl.arange(0, BC)) < T
    last_idx = min((i_t+1) * BT, T) - 1
    o_A = (bos + i_t * BT + tl.arange(0, BC)) * H*BT + i_h * BT
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_a = tl.make_block_ptr(a + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(b + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_ge = tl.make_block_ptr(ge + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_g_last = gi + (bos * H + i_h) * K + last_idx * H * K + tl.arange(0, BK)
    b_g_last = tl.load(p_g_last, mask=m_k, other=0)
    p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_ag = tl.make_block_ptr(ag + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_bg = tl.make_block_ptr(bg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_a = tl.load(p_a, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_gi = tl.load(p_gi, boundary_check=(0, 1)).to(tl.float32)
    b_ge = tl.load(p_ge, boundary_check=(0, 1)).to(tl.float32)

    # deal with decay term.
    g_exp = exp(b_gi)
    g_exp_inv = exp(-b_gi + b_g_last[None, :])
    b_qg = b_q * g_exp
    b_kg = b_k * g_exp_inv
    b_bg = b_b * g_exp_inv
    b_ag = b_a * exp(b_ge)
    tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_bg, b_bg.to(p_bg.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ag, b_ag.to(p_ag.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    # tl.debug_barrier()

    b_q = b_q.to(b_k.dtype)
    # inner attn
    for j in range(0, min(BC, T - i_t * BT)):
        # a trick to index the j-th row of b_k, b_g, b_b
        if GATHER_SUPPORTED:
            row_idx = tl.full([1, BK], j, dtype=tl.int16)
            # [1, BK]
            b_k_j = gather(b_k, row_idx, axis=0)
            b_gk_j = gather(b_gi, row_idx, axis=0)
            b_b_j = gather(b_b, row_idx, axis=0)
        else:
            mask = tl.arange(0, BC) == j
            b_k_j = tl.sum(tl.where(mask[:, None], b_k, 0), 0)[None, :]
            b_gk_j = tl.sum(tl.where(mask[:, None], b_gi, 0), 0)[None, :]
            b_b_j = tl.sum(tl.where(mask[:, None], b_b, 0), 0)[None, :]
        tmp = exp(b_gi - b_gk_j)
        b_A_qk = tl.sum(b_q * b_k_j * tmp, 1)
        m_i = (o_i >= j).to(tl.float32)
        b_A_qk = b_A_qk * m_i
        b_A_qb = tl.sum(b_q * b_b_j * tmp, 1)
        b_A_qb = b_A_qb * m_i
        tmp2 = exp(b_ge - b_gk_j)
        b_A_ak = tl.sum(b_a * b_k_j * tmp2, 1)
        m_i2 = (o_i > j).to(tl.float32)
        b_A_ak = b_A_ak * m_i2
        b_A_ab = tl.sum(b_a * b_b_j * tmp2, 1)
        b_A_ab = b_A_ab * m_i2

        tl.store(Aqk + o_A + j, b_A_qk.to(dtype=Aqk.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_A)
        tl.store(Aqb + o_A + j, b_A_qb.to(dtype=Aqb.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_A)
        tl.store(Aab + o_A + j, b_A_ab.to(dtype=Aqb.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_A)
        tl.store(Aak + o_A + j, b_A_ak.to(dtype=Aqk.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_A)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=['BK', 'BT'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_A_kernel_intra_tensorcore(
    q,
    k,
    a,
    b,
    gi,
    ge,
    qg,
    kg,
    ag,
    bg,
    Aqk,
    Aqb,
    Aab,
    Aak,
    cu_seqlens,
    chunk_indices,
    scale: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr,
):
    i_t, i_b, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_len = eos - bos
    else:
        bos = i_b * T
        T_len = T

    if i_t * BT >= T_len:
        return

    # Compute base offset for all tensors
    offset_base = (bos * H + i_h) * K

    # Load the current chunk of Q, K, A, B and their gates
    p_q = tl.make_block_ptr(q + offset_base, (T_len, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + offset_base, (T_len, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_a = tl.make_block_ptr(a + offset_base, (T_len, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_b = tl.make_block_ptr(b + offset_base, (T_len, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi + offset_base, (T_len, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_ge = tl.make_block_ptr(ge + offset_base, (T_len, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_a = tl.load(p_a, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_gi_val = tl.load(p_gi, boundary_check=(0, 1)).to(tl.float32)
    b_ge_val = tl.load(p_ge, boundary_check=(0, 1)).to(tl.float32)

    # Calculate the index of the middle element of the valid part of the chunk
    valid_len = min(T_len - i_t * BT, BT)
    mid_idx = valid_len // 2

    # Load the offset vector from Global Memory
    # p_offset points to gi[i_t*BT + mid_idx, :]
    m_k = tl.arange(0, BK) < K
    p_offset = gi + offset_base + (i_t * BT + mid_idx) * K + tl.arange(0, BK)
    b_offset = tl.load(p_offset, mask=m_k, other=0.0).to(tl.float32)

    # Apply offset to gate values
    # These operations broadcast [BK] to [BT, BK]
    b_gi_val = b_gi_val - b_offset[None, :]
    b_ge_val = b_ge_val - b_offset[None, :]

    # Apply decay factors (now numerically safe)
    # q_term ~ exp(gi - offset)
    # k_term ~ exp(-gi + offset)
    exp_gi = tl.exp(b_gi_val)
    inv_exp_gi = tl.exp(-b_gi_val)
    exp_ge = tl.exp(b_ge_val)

    b_q = (b_q * scale).to(tl.float32)

    # Compute gated operands for matrix multiplication
    # q, k in l2 norm, < 1, i
    q_ops = (b_q * exp_gi).to(tl.float32)
    k_ops = (b_k * inv_exp_gi).to(tl.float32)
    b_ops = (b_b * inv_exp_gi).to(tl.float32)
    a_ops = (b_a * exp_ge).to(tl.float32)

    # Load gate values at the last position for inter-chunk decay
    last_idx = min((i_t+1) * BT, T_len) - 1
    p_g_last = gi + offset_base + last_idx * H * K + tl.arange(0, BK)
    b_g_last = tl.load(p_g_last, mask=m_k, other=0.0)

    exp_offset = tl.exp(b_offset)
    b_g_centered = b_g_last - b_offset
    exp_g_centered = tl.exp(b_g_centered)

    # Create pointers for writing
    p_qg = tl.make_block_ptr(qg + offset_base, (T_len, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_kg = tl.make_block_ptr(kg + offset_base, (T_len, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_ag = tl.make_block_ptr(ag + offset_base, (T_len, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_bg = tl.make_block_ptr(bg + offset_base, (T_len, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))

    # Store gated Q and A
    tl.store(p_qg, (q_ops * exp_offset[None, :]).to(p_qg.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_ag, (a_ops * exp_offset[None, :]).to(p_ag.dtype.element_ty), boundary_check=(0, 1))

    # Store gated K and B
    b_kg_g = k_ops * exp_g_centered[None, :]
    b_bg_g = b_ops * exp_g_centered[None, :]
    tl.store(p_kg, b_kg_g.to(p_kg.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_bg, b_bg_g.to(p_bg.dtype.element_ty), boundary_check=(0, 1))

    # Transpose K and B for dot product
    k_ops_t = tl.trans(k_ops)
    b_ops_t = tl.trans(b_ops)

    # Compute intra-chunk attention using TensorCores
    q_ops_h = q_ops.to(b_q.dtype)
    b_A_qk = tl.dot(q_ops_h, k_ops_t.to(b_q.dtype))
    b_A_qb = tl.dot(q_ops_h, b_ops_t.to(b_q.dtype))
    b_A_ak = tl.dot(a_ops, k_ops_t)
    b_A_ab = tl.dot(a_ops, b_ops_t)

    # Create causal masks
    offs_n = tl.arange(0, BT)
    offs_m = tl.arange(0, BT)
    mask_inclusive = offs_m[:, None] >= offs_n[None, :]
    mask_strict = offs_m[:, None] > offs_n[None, :]

    # Apply causal masking
    b_A_qk = tl.where(mask_inclusive, b_A_qk, 0.0)
    b_A_qb = tl.where(mask_inclusive, b_A_qb, 0.0)
    b_A_ak = tl.where(mask_strict, b_A_ak, 0.0)
    b_A_ab = tl.where(mask_strict, b_A_ab, 0.0)

    # Store the intra-chunk attention matrices
    offset_out_base = (bos * H + i_h) * BT

    p_Aqk = tl.make_block_ptr(Aqk + offset_out_base, (T_len, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_Aqb = tl.make_block_ptr(Aqb + offset_out_base, (T_len, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_Aak = tl.make_block_ptr(Aak + offset_out_base, (T_len, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_Aab = tl.make_block_ptr(Aab + offset_out_base, (T_len, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    tl.store(p_Aqk, b_A_qk.to(p_Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqb, b_A_qb.to(p_Aqb.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aak, b_A_ak.to(p_Aak.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aab, b_A_ab.to(p_Aab.dtype.element_ty), boundary_check=(0, 1))


def chunk_dplr_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    scale: float,
    chunk_size: int,
    safe_gate: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K = k.shape
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    Aqk = q.new_empty(B, T, H, BT, dtype=q.dtype)
    Aqb = q.new_empty(B, T, H, BT, dtype=q.dtype)
    # involving matrix inverse and it'd be better to use float here.
    Aab = q.new_empty(B, T, H, BT, dtype=torch.float)
    Aak = q.new_empty(B, T, H, BT, dtype=torch.float)

    grid = (NT, B, H)
    BK = max(triton.next_power_of_2(K), 16)
    qg = torch.empty_like(q)
    kg = torch.empty_like(k, dtype=q.dtype)
    ag = torch.empty_like(a, dtype=q.dtype)
    bg = torch.empty_like(b, dtype=q.dtype)
    if safe_gate:
        chunk_dplr_fwd_A_kernel_intra_func = chunk_dplr_fwd_A_kernel_intra_tensorcore
    else:
        chunk_dplr_fwd_A_kernel_intra_func = chunk_dplr_fwd_A_kernel_intra_sub_intra
    chunk_dplr_fwd_A_kernel_intra_func[grid](
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        Aqk=Aqk,
        Aqb=Aqb,
        Aab=Aab,
        Aak=Aak,
        qg=qg,
        kg=kg,
        ag=ag,
        bg=bg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BT,
        BK=BK,
        GATHER_SUPPORTED=IS_GATHER_SUPPORTED,
    )
    return Aab, Aqk, Aak, Aqb, qg, kg, ag, bg
