# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp, exp2
from fla.utils import IS_NVIDIA_BLACKWELL, autotune_cache_kwargs, check_shared_mem

if IS_NVIDIA_BLACKWELL:
    """
    Compute tl.dot with SM100 workaround.

    On SM100 (Blackwell) GPUs, wraps the result in inline assembly to prevent
    the TritonGPUHoistTMEMAlloc pass from incorrectly fusing add and dot operations.
    See: https://github.com/fla-org/flash-linear-attention/issues/638

    TODO: Remove this workaround once the Triton compiler bug is fixed.
    Track upstream issue at: https://github.com/triton-lang/triton/issues/8695
    """
    @triton.jit
    def safe_dot(a, b):
        return tl.inline_asm_elementwise(
            asm="mov.f32 $0, $1;",
            constraints="=r,r",
            args=[tl.dot(a, b)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
else:
    @triton.jit
    def safe_dot(a, b):
        return tl.dot(a, b)


# ==============================================================================
# Asymmetric WY Backward Kernel for Preconditioned Gated Delta Rule
# ==============================================================================
# Key difference from symmetric version:
# - Forward KKT: baseKKT_ij = beta_i * <k_i, k_precond_j>
# - Backward must use: b_A = (beta*k) @ k_precond^T
# - Row-side gradients -> dk
# - Column-side gradients -> dk_precond (separate output for ATK backward)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_precond_wy_repr_bwd_kernel(
    k,           # [B, T, H, K] - original k (READ key, row side)
    k_precond,   # [B, T, H, K] - preconditioned k (WRITE key, column side)
    v,           # [B, T, HV, V]
    beta,        # [B, T, HV]
    g,           # [B, T, HV] - gate cumsum
    A,           # [B, T, HV, BT] - inverse WY matrix
    dw,          # [B, T, HV, K] - gradient of w
    du,          # [B, T, HV, V] - gradient of u
    dk,          # [B, T, HV, K] - output: gradient w.r.t. original k (row side)
    dk_precond,  # [B, T, HV, K] - output: gradient w.r.t. k_precond (column side)
    dv,          # [B, T, HV, V] - output: gradient w.r.t. v
    db,          # [B, T, HV] - output: gradient w.r.t. beta
    dg,          # [B, T, HV] - output: gradient w.r.t. g (cumsum)
    cu_seqlens,     # *i32 [N+1] - cumulative sequence lengths
    chunk_indices,  # *i32 [NT, 2] - (seq_idx, chunk_idx) pairs
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Asymmetric WY backward for preconditioned gated delta rule.

    Key insight: The forward uses asymmetric KKT matrix:
        baseKKT_ij = beta_i * <k_i, k_precond_j>

    Row index (i) uses original k (READ key)
    Column index (j) uses k_precond (WRITE key)

    Gradient flow:
    - Row-side (from dA @ K_precond): goes to dk and dbeta
    - Column-side (from dA^T @ (beta*k)): goes to dk_precond only
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_r = tl.arange(0, BT)

    b_b = tl.load(beta + (bos*HV + i_hv) + o_t*HV, mask=m_t, other=0.0)
    b_db = tl.zeros([BT], dtype=tl.float32)
    p_A = A + (bos*HV + i_hv) * BT + o_r[:, None] + o_t[None, :] * (HV*BT)
    b_A = tl.load(p_A, mask=(o_r[:, None] < BT) & m_t[None, :], other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_G:
        b_g = tl.load(g + (bos*HV + i_hv) + o_t*HV, mask=m_t, other=0.0)
        if USE_EXP2:
            b_g_exp = exp2(b_g)
        else:
            b_g_exp = tl.exp(b_g)
        b_dg = tl.zeros([BT], dtype=tl.float32)

    # First pass: accumulate dA from dw using original k (for w = A^-1 @ (k * beta * g))
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_dk = dk + (bos*HV + i_hv) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        p_dw = dw + (bos*HV + i_hv) * K + o_t[:, None] * (HV*K) + o_k[None, :]

        # [BT, BK]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        if USE_G:
            b_kbg = b_k * (b_b * b_g_exp)[:, None]
        else:
            b_kbg = b_k * b_b[:, None]
        b_dw = tl.load(p_dw, mask=m_tk, other=0.0)

        # dA contribution from dw: dA += dw @ kbg^T
        b_dA += safe_dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))

        # dk contribution from first term: A @ dw
        b_dkbg = safe_dot(b_A, b_dw.to(b_A.dtype))
        if USE_G:
            b_dk = b_dkbg * (b_g_exp * b_b)[:, None]
            b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], 1)
            b_dg += tl.sum(b_dkbg * b_kbg, 1)
        else:
            b_dk = b_dkbg * b_b[:, None]
            b_db += tl.sum(b_dkbg * b_k, 1)

        tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_tk)

    # Process dv and du (u = A^-1 @ (v * beta), same as symmetric)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*HV + i_hv) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        p_dv = dv + (bos*HV + i_hv) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        p_du = du + (bos*HV + i_hv) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, mask=m_tv, other=0.0)
        b_dA += safe_dot(b_du, tl.trans(b_vb))
        b_dvb = safe_dot(b_A, b_du.to(b_A.dtype))
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv.to(dv.dtype.element_ty), mask=m_tv)

    # Transform dA through lower triangular inversion
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = safe_dot(b_dA.to(b_A.dtype), b_A)
    b_dA = safe_dot(b_A, b_dA.to(b_A.dtype))

    if USE_G:
        if USE_EXP2:
            b_dA *= exp2(b_g[:, None] - b_g[None, :])
        else:
            b_dA *= exp(b_g[:, None] - b_g[None, :])

    b_dA = tl.where(m_A, -b_dA, 0)

    b_dA = b_dA.to(k.dtype.element_ty)

    tl.debug_barrier()
    # ==== ASYMMETRIC PART ====
    # Reconstruct b_A (the asymmetric KKT) and compute gradients
    # baseKKT_ij = beta_i * <k_i, k_precond_j>
    # Row index i: uses k (READ key)
    # Column index j: uses k_precond (WRITE key)
    #
    # From the gradient chain rule:
    # - dL/dk_i += sum_j dA_ij * k_precond_j * beta_i  (row-side)
    # - dL/dk_precond_j += sum_i dA_ij * k_i * beta_i  (column-side)
    # - dL/dbeta_i += sum_j dA_ij * <k_i, k_precond_j>

    b_A_asymm = tl.zeros([BT, BT], dtype=tl.float32)  # Asymmetric KKT matrix

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_kp = k_precond + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_dk = dk + (bos*HV + i_hv) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        p_dkp = dk_precond + (bos*HV + i_hv) * K + o_t[:, None] * (HV*K) + o_k[None, :]

        b_k = tl.load(p_k, mask=m_tk, other=0.0)   # [BT, BK] - original k
        b_kp = tl.load(p_kp, mask=m_tk, other=0.0)  # [BT, BK] - k_precond
        b_dk = tl.load(p_dk, mask=m_tk, other=0.0)  # Load existing dk from first pass

        # kb = k * beta (used on row side)
        b_kb = (b_k * b_b[:, None]).to(b_k.dtype)

        # Build asymmetric KKT: (beta * k) @ k_precond^T
        b_A_asymm += safe_dot(b_kb, tl.trans(b_kp))

        # Row-side gradient: dkb = dA @ k_precond
        b_dkb = safe_dot(b_dA, b_kp)
        # dk contribution: dkb * beta
        b_dk += b_dkb * b_b[:, None]
        # dbeta contribution: <dkb, k>
        b_db += tl.sum(b_dkb * b_k, 1)

        # Column-side gradient: dk_precond = dA^T @ (beta * k)
        b_dkp = safe_dot(tl.trans(b_dA), b_kb)

        tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_tk)
        tl.store(p_dkp, b_dkp.to(dk_precond.dtype.element_ty), mask=m_tk)

    tl.store(db + (bos*HV + i_hv) + o_t*HV, b_db.to(db.dtype.element_ty), mask=m_t)

    # dg using the asymmetric KKT matrix
    if USE_G:
        b_AdA = b_dA * b_A_asymm
        b_dg += tl.sum(b_AdA, axis=1) - tl.sum(b_AdA, axis=0)
        tl.store(dg + (bos*HV + i_hv) + o_t*HV, b_dg.to(dg.dtype.element_ty), mask=m_t)


def prepare_precond_wy_repr_bwd(
    k: torch.Tensor,
    k_precond: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric WY backward for preconditioned gated delta rule.

    Unlike the symmetric version that uses the same k for both row and column
    of the KKT matrix, this version uses:
    - k (original) for row side (READ key)
    - k_precond for column side (WRITE key)

    Args:
        k: Original keys [B, T, H, K]
        k_precond: Preconditioned keys [B, T, H, K]
        v: Values [B, T, H, V]
        beta: Beta scaling [B, T, H]
        g: Gate cumsum [B, T, H]
        A: Inverse WY matrix [B, T, H, BT]
        dw: Gradient of w [B, T, H, K]
        du: Gradient of u [B, T, H, V]
        cu_seqlens: Cumulative sequence lengths [N+1] for varlen
        chunk_indices: Precomputed chunk indices [NT, 2] for varlen

    Returns:
        dk: Gradient w.r.t. original k (row-side) [B, T, H, K]
        dk_precond: Gradient w.r.t. k_precond (column-side) [B, T, H, K]
        dv: Gradient w.r.t. v [B, T, H, V]
        dbeta: Gradient w.r.t. beta [B, T, H]
        dg: Gradient w.r.t. g cumsum [B, T, H]
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BT = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dk = k.new_empty(B, T, HV, K)
    dk_precond = k.new_empty(B, T, HV, K)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g) if g is not None else None
    db = torch.empty_like(beta)

    prepare_precond_wy_repr_bwd_kernel[(NT, B * HV)](
        k=k,
        k_precond=k_precond,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dk_precond=dk_precond,
        dv=dv,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_EXP2=use_exp2,
    )

    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
        dk_precond = dk_precond.view(B, T, H, HV // H, K).sum(3)

    return dk, dk_precond, dv, db, dg
