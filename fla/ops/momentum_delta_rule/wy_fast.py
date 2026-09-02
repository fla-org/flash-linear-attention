# Copyright (c) 2023-2025,


import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.utils import autotune_cache_kwargs, check_shared_mem

NUM_WARPS = [2, 4, 8]
NUM_WARPS = [2, 4]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [16, 32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_mode_rule_pkt_fwd_kernel(
    k,
    p,
    log_a_cum,
    log_mu_cum,
    log_ct,
    A,
    gamma_mask_q,
    bt,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # notation:
    # b_ means data
    # p_ means pointer
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # prepare mask
    o_t = i_t * BT + tl.arange(0, BT)   # The index of inside the chunk BT
    m_t = o_t < T   # mask
    # mask_A    = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)   # tril mask
    mask_tril = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)   # tril mask

    i, j = tl.arange(0, BT)[:, None], tl.arange(0, BT)[None, :]
    S_m = tl.where(i == (j + 1), 1.0, 0.0)                   # [BT, BT]

    # offsets for 1D log tensors: shape (T,) stride H, base = log_* + bos*H + i_h
    p_log_acum_base = log_a_cum + bos * H + i_h
    p_log_mcum_base = log_mu_cum + bos * H + i_h
    p_log_ct_base = log_ct + bos * H + i_h

    b_log_a_cum = tl.load(p_log_acum_base + o_t * H, mask=m_t, other=0.0)  # [BT]
    b_log_m_cum = tl.load(p_log_mcum_base + o_t * H, mask=m_t, other=0.0)  # [BT]
    b_log_ct = tl.load(p_log_ct_base + o_t * H, mask=m_t, other=0.0)  # [BT]

    # c_{t}
    # b_ct = tl.exp(b_log_ct)
    r = tl.arange(0, BT)
    # \bar{a}_{t-1}

    # c_{t-1}
    neg_inf = tl.zeros([1], dtype=tl.float32) - float("inf")
    b_log_c_tm1 = tl.where(r == 0, neg_inf, tl.sum(S_m * b_log_ct[None, :], axis=1))
    b_bt = tl.exp(b_log_a_cum + b_log_ct)    # b_t

    a = b_log_ct[:, None]
    b = b_log_c_tm1[None, :]
    x = b - a                    # <=0 du
    x = 1 - tl.exp(x)

    b_log_gamma = tl.where(mask_tril, (b_log_a_cum + b_log_ct)[:, None] - b_log_m_cum[None, :], 0)
    b_gamma_mask_q = tl.where(mask_tril, tl.exp(b_log_gamma) * x, 0)

    b_gamma_mask = tl.dot(S_m, b_gamma_mask_q)               # first row is zero
    b_gamma_mask = tl.where(mask_tril, b_gamma_mask, 0.0)    # strict tril

    # p * k
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        # k base: (K, T) with strides (1, H*K), base = k + (bos*H + i_h)*K
        # p base: (T, K) with strides (H*K, 1), base = p + (bos*H + i_h)*K
        base_k = k + (bos * H + i_h) * K
        base_p = p + (bos * H + i_h) * K
        m_k_t = (o_k[:, None] < K) & (o_t[None, :] < T)
        m_t_k = (o_t[:, None] < T) & (o_k[None, :] < K)
        p_k = base_k + o_k[:, None] * 1 + o_t[None, :] * (H * K)
        p_p = base_p + o_t[:, None] * (H * K) + o_k[None, :] * 1

        b_k = tl.load(p_k, mask=m_k_t, other=0.0)
        b_p = tl.load(p_p, mask=m_t_k, other=0.0)
        b_A += tl.dot(b_p, b_k)

    b_A = b_A * b_gamma_mask

    # store results: (T, BT) with strides (BT*H, 1), base = A + (bos*H + i_h)*BT
    o_bt = tl.arange(0, BT)
    base_A = A + (bos * H + i_h) * BT
    base_gamma = gamma_mask_q + (bos * H + i_h) * BT
    m_A = (o_t[:, None] < T) & (o_bt[None, :] < BT)
    p_A = base_A + o_t[:, None] * (BT * H) + o_bt[None, :] * 1
    p_gamma_q = base_gamma + o_t[:, None] * (BT * H) + o_bt[None, :] * 1
    # bt: (T,) stride H
    p_bt_base = bt + bos * H + i_h
    p_bt = p_bt_base + o_t * H

    tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_A)
    tl.store(p_gamma_q, b_gamma_mask_q.to(p_gamma_q.dtype.element_ty), mask=m_A)
    tl.store(p_bt, b_bt.to(p_bt.dtype.element_ty), mask=m_t)


def chunk_scaled_dot_mode_rule_pkt_fwd(
    k: torch.Tensor,
    p: torch.Tensor,
    log_a_cum: torch.Tensor | None = None,
    log_mu_cum: torch.Tensor | None = None,
    log_ct: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Compute beta \mathcal{A}(i-1/j) * P * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        p (torch.Tensor):
            The auxiliary key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g0 (torch.Tensor):
            The cumulative sum minus the original one of the gate tensor of shape `[B, T, H]`.
            Default: None
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`.
            Default: None
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # need solve reverse
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    # for computing readout
    gamma_mask_q = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

    bt = torch.empty(B, T, H, device=k.device, dtype=output_dtype)

    chunk_scaled_dot_mode_rule_pkt_fwd_kernel[(NT, B * H)](
        k=k,
        p=p,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        log_ct=log_ct,
        A=A,
        gamma_mask_q=gamma_mask_q,
        bt=bt,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )
    return A, bt, gamma_mask_q


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        # for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_uyz_repr_bwd_kernel(
    q,
    k,
    v,
    p,
    beta,
    log_a_cum,
    log_m_cum,
    gamma_mask_q,
    d_Attn_do_v,
    d_decay_s,
    bt,
    log_ct,
    A,
    du,
    dy,
    dz,
    dk,
    dv,
    dp,
    dbt,
    dlog_a,
    dlog_mu,
    d_log_mu_cum,
    d_log_a_cum,
    dbeta,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    p += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dp += (bos * H + i_h) * K
    dy += (bos * H + i_h) * K
    dz += (bos * H + i_h) * K

    v += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    du += (bos * H + i_h) * V

    bt += bos * H + i_h
    log_a_cum += bos * H + i_h
    log_m_cum += bos * H + i_h
    beta += bos * H + i_h
    log_ct += bos * H + i_h
    dlog_a += bos * H + i_h
    dlog_mu += bos * H + i_h
    dbeta += bos * H + i_h
    dbt += bos * H + i_h
    d_log_mu_cum += bos * H + i_h
    d_log_a_cum += bos * H + i_h

    d_decay_s += bos * H + i_h

    A += (bos * H + i_h) * BT
    gamma_mask_q += (bos * H + i_h) * BT
    d_Attn_do_v += (bos * H + i_h) * BT

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_bt = tl.arange(0, BT)

    # 1D loads: (T,) stride H
    b_log_a_cum = tl.load(log_a_cum + o_t * H, mask=m_t, other=0.0)  # [BT]
    b_log_m_cum = tl.load(log_m_cum + o_t * H, mask=m_t, other=0.0)  # [BT]
    b_bt = tl.load(bt + o_t * H, mask=m_t, other=0.0)  # [BT]

    # A: (BT, T) with strides (1, H*BT) -> base + o_row*1 + o_t*(H*BT)
    o_A_row = tl.arange(0, BT)
    m_A_T = (o_A_row[:, None] < BT) & (o_t[None, :] < T)
    p_A = A + o_A_row[:, None] * 1 + o_t[None, :] * (H * BT)
    b_A = tl.load(p_A, mask=m_A_T, other=0.0)  # [BT, BT]

    # gamma_mask_q: (T, BT) with strides (H*BT, 1) -> base + o_t*(H*BT) + o_bt
    m_gamma = (o_t[:, None] < T) & (o_bt[None, :] < BT)
    p_gamma_q = gamma_mask_q + o_t[:, None] * (H * BT) + o_bt[None, :] * 1
    b_gamma_mask_q = tl.load(p_gamma_q, mask=m_gamma, other=0.0)

    # masks for BT x BT computations
    mask_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)   # strict tril
    mask_tril = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)   # tril

    i, j = tl.arange(0, BT)[:, None], tl.arange(0, BT)[None, :]
    S_m1 = tl.where(i == (j + 1), 1.0, 0.0)                                                        # [BT, BT]
    S_p1 = tl.where((i + 1) == j, 1.0, 0.0)                                                        # [BT, BT]
    b_gamma_mask = tl.dot(S_m1, b_gamma_mask_q)
    b_gamma_mask = tl.where(mask_A, b_gamma_mask, 0.0)                          # strict tril

    b_bar_a_tm1 = tl.exp(tl.sum(S_m1 * b_log_a_cum[None, :], axis=1))        # [BT]
    b_b_tm1 = tl.sum(S_m1 * b_bt[None, :], axis=1)                           # [BT]

    b_d_bar_a_tm1 = tl.zeros([BT], dtype=tl.float32)
    b_d_btm1 = tl.zeros([BT], dtype=tl.float32)
    b_d_log_ct = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = (o_t[:, None] < T) & (o_k[None, :] < K)
        p_p = p + o_t[:, None] * (H * K) + o_k[None, :] * 1
        p_dp = dp + o_t[:, None] * (H * K) + o_k[None, :] * 1
        p_dy = dy + o_t[:, None] * (H * K) + o_k[None, :] * 1
        p_dz = dz + o_t[:, None] * (H * K) + o_k[None, :] * 1
        """
        # The 'd' means partial when 'd' in dL/dx
        # for BK part:
        d_bar_a_tm1 = dL/dy * dy/d_bar_a_tm1_p * d_bar_a_tm1_p/d_bar_a_tm1
                       = dy * A * k                                                          [BT, BK] -> [BT, 1]
        d_b_tm1 = dL/dz * dz/d_b_tm1_p * d_b_tm1_p/d_b_tm1
                       = dz * A * k                                                          [BT, BK] -> [BT, 1]
        dp = dL/dy * dy/d_bar_a_tm1_p * d_bar_a_tm1_p/d_p + dL/dz * dz/d_b_tm1_p * d_b_tm1_p/d_p
                       = d_bar_a_tm1 * bar_a_tm1  + d_b_tm1 * b_tm1                                     [BT, BK]
        dL/dAttn_inv = dL/du * du/dAttn_inv + dL/dy * dy/dAttn_inv + dL/dz * dz/dAttn_inv
                   = du * v + (dy * alpha_tm1_p + dz * d_b_tm1_p)                                       [BT, BT]
        """
        b_p = tl.load(p_p, mask=m_tk, other=0.0)
        b_dy = tl.load(p_dy, mask=m_tk, other=0.0)
        b_dz = tl.load(p_dz, mask=m_tk, other=0.0)
        b_dalpha_tm1_p = tl.dot(b_A.to(b_dy.dtype), b_dy)
        b_db_tm1_p = tl.dot(b_A.to(b_dz.dtype), b_dz)                                                                # [BT]
        b_dp = b_dalpha_tm1_p * b_bar_a_tm1[:, None] + b_db_tm1_p * b_b_tm1[:, None]

        b_p_bar_a_tm1 = b_p * b_bar_a_tm1[:, None]
        b_p_b_tm1 = b_p * b_b_tm1[:, None]
        b_dA += tl.dot(b_dy, tl.trans(b_p_bar_a_tm1).to(b_dy.dtype)) + tl.dot(b_dz, tl.trans(b_p_b_tm1).to(b_dz.dtype))
        b_d_bar_a_tm1 += tl.sum(b_dalpha_tm1_p * b_p, axis=1)
        b_d_btm1 += tl.sum(b_db_tm1_p * b_p, axis=1)

        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), mask=m_tk)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = (o_t[:, None] < T) & (o_v[None, :] < V)
        p_v = v + o_t[:, None] * (H * V) + o_v[None, :] * 1
        p_dv = dv + o_t[:, None] * (H * V) + o_v[None, :] * 1
        p_du = du + o_t[:, None] * (H * V) + o_v[None, :] * 1
        """
        # for BV part:
        dv = dL/du * du/dv = du * A                                                                        [BT, BV]
        dL/dAttn_inv = dL/du * du/dAttn_inv + dL/dy * dy/dAttn_inv + dL/dz * dz/dAttn_inv
                       = (du * v) + dy * alpha_tm1_p + dz * d_b_tm1_p                                      [BT, BT]
        """
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        b_du = tl.load(p_du, mask=m_tv, other=0.0)
        b_dv = tl.dot(b_A.to(b_du.dtype), b_du)
        b_dA += tl.dot(b_du, tl.trans(b_v))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_tv)

    """
    from  Attn_inv * A = I
    get   dA = - A * d(Attn_inv) * A                                                                     [BT, BT]
    """
    # o_t/m_t already defined, reuse

    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))      # A * dA_inv
    b_dA = -tl.dot(b_dA.to(b_A.dtype), b_A)      # A * dA_inv * A

    b_dG = b_dA * b_gamma_mask  # b_gamma_mask already tril
    b_G = tl.zeros([BT, BT], dtype=tl.float32)
    b_Gqk = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = (o_t[:, None] < T) & (o_k[None, :] < K)
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :] * 1
        p_k = k + o_t[:, None] * (H * K) + o_k[None, :] * 1
        p_p = p + o_t[:, None] * (H * K) + o_k[None, :] * 1
        p_dk = dk + o_t[:, None] * (H * K) + o_k[None, :] * 1
        p_dp = dp + o_t[:, None] * (H * K) + o_k[None, :] * 1
        """
        # for next BK part
        pkt = (p @ k.transpose(-1, -2))                                                                 [BT, BT]
        d_gamma_mask = dL/dA * dA/dgamma_mask
                       = dA * kkt                                                                       [BT, BT]
        d_pkt = dA * gamma_mask                                                                         [BT, BT]
        d_p = dL/d_kkt * d_kkt/dp = d_kkt^T * k                                                         [BT, BK]
        d_k = dL/d_kkt * d_kkt/dk = d_kkt^T * p                                                         [BT, BK]
        """
        b_q = tl.load(p_q, mask=m_tk, other=0.0)
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_p = tl.load(p_p, mask=m_tk, other=0.0)
        b_dp = tl.load(p_dp, mask=m_tk, other=0.0)

        b_G += tl.dot(b_p, tl.trans(b_k))
        b_Gqk += tl.dot(b_q, tl.trans(b_k))
        b_dp += tl.dot(b_dG.to(b_k.dtype), b_k)
        b_dk = tl.dot(tl.trans(b_dG).to(b_k.dtype), b_p)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)
        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), mask=m_tk)

    b_d_bt = tl.load(dbt + o_t * H, mask=m_t, other=0.0)  # [BT]
    b_log_ct = tl.load(log_ct + o_t * H, mask=m_t, other=0.0)  # [BT]
    b_beta = tl.load(beta + o_t * H, mask=m_t, other=0.0)  # [BT]
    b_d_log_mu_cum = tl.load(d_log_mu_cum + o_t * H, mask=m_t, other=0.0)  # [BT]
    b_d_log_a_cum = tl.load(d_log_a_cum + o_t * H, mask=m_t, other=0.0)  # [BT]

    """
    # dL/dgamma_mask_q = dL/dAttn * dAttn/dgamma_mask_q + dL/d_decay_s
    #                         = dL/dAttn * q * k                                            [BT, BT]
    """
    # d_Attn_do_v: (T, BT) with strides (H*BT, 1)
    p_dAttn_do_v = d_Attn_do_v + o_t[:, None] * (H * BT) + o_bt[None, :] * 1
    b_dAttn_do_v = tl.load(p_dAttn_do_v, mask=m_gamma, other=0.0)
    b_d_decay_s = tl.load(d_decay_s + o_t * H, mask=m_t, other=0.0)

    b_dgamma_mask_q = b_dAttn_do_v * b_Gqk * scale                       # [BT, BT]

    rel_last = (min((i_t + 1) * BT, T) - 1) - i_t * BT  # scalar in [0, BT-1]
    rows = tl.arange(0, BT)[:, None]  # [BT, 1]
    # last_mask = rows == rel_last
    b_last_row = tl.where(rows < rel_last, 0.0, b_d_decay_s[None, :])  # [BT, BT]
    b_dgamma_mask_q += b_last_row   # b_d_gamma_mask_q_last = b_ddecay_s

    b_dgamma_mask = b_dA * b_G                          # dL/dgamma = dA * G
    b_dgamma_mask_q += tl.dot(S_p1, b_dgamma_mask)                                                   # [BT, BT]

    NEG_INF = tl.zeros([1], dtype=tl.float32) - float("inf")
    b_log_c_tm1 = tl.sum(S_m1 * b_log_ct[None, :], axis=1)
    r = tl.arange(0, BT)
    b_log_c_tm1 = tl.where(r == 0, NEG_INF, b_log_c_tm1)
    b_bt = tl.exp(b_log_a_cum + b_log_ct)    # b_t

    # according to the:
    #     gamma_mask_q = (log_a_cum.unsqueeze(-1) - log_m_cum.unsqueeze(-2) + log_c_jt).exp().float().tril()
    #     gamma_mask = torch.cat([torch.zeros_like(gamma_mask_q[:, :, :, :1]), gamma_mask_q[:, :, :, :-1]], dim=3)
    b_d_log_gamma = tl.where(mask_tril, b_dgamma_mask_q * b_gamma_mask_q, 0.0)                       # [BT, BT]

    b_d_log_a_cum += tl.sum(b_d_log_gamma, axis=1)      # row sum                                     # [BT]
    b_d_log_mu_cum -= tl.sum(b_d_log_gamma, axis=0)      # col sum                                      # [BT]
    b_d_log_c_jt = b_d_log_gamma

    # according to the:
    #     b_t   = (log_a_cum + log_ct).exp()   # b_t
    #     b_tm1 = torch.cat([torch.zeros_like(b_t[:, :, :, :1]), b_t[:, :, :, :-1]], dim=3)  # b_{t-1}
    b_d_bt += tl.sum(S_p1 * b_d_btm1[None, :], axis=1)

    temp = b_d_bt * b_bt
    b_d_log_a_cum += temp
    b_d_log_ct += temp

    # according to:
    #     log_bar_a_tm1 = torch.cat([torch.zeros_like(log_a_cum[:, :, :, :1]), log_a_cum[:, :, :, :-1]], dim=3)
    #     bar_a_tm1 = log_bar_a_tm1.exp()                                                    # \bar{a}_{t-1}
    b_dlog_bar_a_tm1 = b_d_bar_a_tm1 * b_bar_a_tm1
    b_d_log_a_cum += tl.sum(S_p1 * b_dlog_bar_a_tm1[None, :], axis=1)

    # according to:
    #     log_ct_tm1 = torch.cat([torch.full_like(log_ct[:, :, :, :1], float('-inf')), log_ct[:, :, :, :-1]], dim=-1)
    #     a = log_ct.unsqueeze(-1)
    #     b = log_ct_tm1.unsqueeze(-2)
    #     x = (b - a).tril()                          # x <= 0
    #     log_c_jt = a + torch.log(1 - torch.exp(x))   # a + log(1 - exp(x))
    # b_x = tl.where(mask_tril, b_log_c_tm1[None, :] - b_log_ct[:, None], 0.0)
    eps = tl.zeros([1], dtype=tl.float32) + 1e-6

    b_x = b_log_c_tm1[None, :] - b_log_ct[:, None]  # <= 0
    b_x = 1 - tl.exp(b_x)
    b_d_x = tl.where(mask_tril, - b_d_log_c_jt * (1 - b_x) / (b_x + eps), 0.0)

    b_d_log_ct += tl.sum(b_d_log_c_jt - b_d_x, axis=1)
    b_d_b = tl.sum(b_d_x, axis=0)
    b_d_log_ct += tl.sum(S_p1 * b_d_b[None, :], axis=1)

    # according to:
    #     log_c_before = log_beta + log_m_cum - log_a_cum
    #     log_ct = torch.logcumsumexp(log_c_before, dim=-1)                   # \sum _{j=1}^{t} c_j
    # b_d_ct = b_d_log_ct * tl.exp(-b_log_ct)
    b_log_c_before = tl.log(b_beta + eps) + b_log_m_cum - b_log_a_cum  # todo: bug here

    # # 2) w[t, j] = exp(log_c_j - log_ct_t)
    # #    = exp_shift[t, j] / ct[t]
    # weight = b_exp_shift / b_ct[:, None]                                      # [BT, BT]

    # # 3)d_log_c[j] = sum_t g[t] * w[t, j]
    i = tl.arange(0, BT)[:, None]
    j = tl.arange(0, BT)[None, :]
    mask = i >= j

    log_expo = tl.where(mask, b_log_c_before[None, :] - b_log_ct[:, None], NEG_INF)  # [BT, BT]
    col_max = tl.maximum(tl.max(log_expo, axis=1), 0.)
    b_d_log_c_before = tl.sum(tl.exp(log_expo - col_max[None, :]) * b_d_log_ct[:, None], axis=0) * tl.exp(col_max)
    """
    # b_d_log_c_before = b_d_cumsum * tl.exp(b_log_c_before)
    # b_d_log_mu_cum += b_d_log_c_before
    # b_d_log_a_cum  += -b_d_log_c_before
    # b_dlog_alpha = tl.cumsum(b_d_log_a_cum,  axis=0, reverse=True)
    # b_dlog_mu    = tl.cumsum(b_d_log_mu_cum, axis=0, reverse=True)
    # b_d_beta = tl.where(m_t, b_d_log_c_before / b_beta, 0.0)
    """
    b_d_beta = b_d_log_c_before / (b_beta + eps)
    b_d_log_mu_cum += b_d_log_c_before
    b_d_log_a_cum += -b_d_log_c_before
    b_dlog_alpha = tl.cumsum(b_d_log_a_cum, axis=0, reverse=True)
    b_dlog_mu = tl.cumsum(b_d_log_mu_cum, axis=0, reverse=True)

    tl.store(dbeta + o_t * H, b_d_beta.to(dbeta.dtype.element_ty), mask=m_t)
    tl.store(dlog_a + o_t * H, b_dlog_alpha.to(dlog_a.dtype.element_ty), mask=m_t)
    tl.store(dlog_mu + o_t * H, b_dlog_mu.to(dlog_mu.dtype.element_ty), mask=m_t)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        # for num_warps in [2, 4, 8]
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_u_y_z_fwd_kernel(
    p,
    v,
    A,
    log_a_cum,
    bt,
    u,
    y,
    z,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr
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
    m_t = o_t < T
    o_bt = tl.arange(0, BT)
    m_A = (o_t[:, None] < T) & (o_bt[None, :] < BT)

    b_log_a_cum = tl.load(log_a_cum + bos * H + i_h + o_t * H, mask=m_t, other=0.0)  # [BT]
    b_bt = tl.load(bt + bos * H + i_h + o_t * H, mask=m_t, other=0.0)             # [BT]
    b_A = tl.load(A + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_bt[None, :], mask=m_A, other=0.0)

    i, j = tl.arange(0, BT)[:, None], tl.arange(0, BT)[None, :]
    S_m1 = tl.where(i == (j + 1), 1.0, 0.0)

    """
    # b_log_bar_a_tm1 = tl.sum(S_m * b_log_a_cum[None, :], axis=1)
    # is_first_chunk = (i_t == 0)
    # b_bar_a_tm1 = tl.exp(tl.where(r == 0, 0.0, b_log_bar_a_tm1))


    # NEG_INF = tl.zeros([1], dtype=tl.float32) - float("inf")
    # b_log_c_tm1 = tl.sum(S_m1 * b_log_ct[None, :], axis=1)
    # r = tl.arange(0, BT)
    # b_log_c_tm1 = tl.where(r == 0, NEG_INF, b_log_c_tm1)

    # c_{t-1}
    neg_inf = tl.zeros([1], dtype=tl.float32) - float("inf")
    b_log_c_tm1 = tl.where(r==0, neg_inf, tl.sum(S_m * b_log_ct[None, :], axis=1))
    # b_log_c_tm1 = tl.where(r==0, neg_inf, tl.load(p_log_c_tm1).to(tl.float32))
    # b_ctm1 = tl.exp(b_log_c_tm1)
    """
    b_bar_a_tm1 = tl.exp(tl.sum(S_m1 * b_log_a_cum[None, :], axis=1))        # [BT]
    b_b_tm1 = tl.sum(S_m1 * b_bt[None, :], axis=1)                           # [BT]

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = (o_t[:, None] < T) & (o_v[None, :] < V)
        p_v = v + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_u = u + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]

        b_v = tl.load(p_v, mask=m_tv, other=0.0)     # [BT, BV]
        b_u = tl.dot(b_A, b_v)                # [BT, BT] @ [BT, BV] -> [BT, BV]
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_tv)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = (o_t[:, None] < T) & (o_k[None, :] < K)
        p_k = p + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :] * 1
        p_y = y + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :] * 1
        p_z = z + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :] * 1

        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_kbara = b_bar_a_tm1[:, None] * b_k
        b_kbtm1 = b_b_tm1[:, None] * b_k

        b_y = tl.dot(b_A, b_kbara.to(b_A.dtype))
        b_z = tl.dot(b_A, b_kbtm1.to(b_A.dtype))

        tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_tk)
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), mask=m_tk)


def recompute_u_y_z_fwd(
    p: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    log_a_cum: torch.Tensor,
    bt: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *p.shape, v.shape[-1]
    BT = A.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 64

    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    u = torch.empty_like(v)
    y = torch.empty_like(p)  # pseudo k
    z = torch.empty_like(p)  # pseudo k
    recompute_u_y_z_fwd_kernel[(NT, B*H)](
        p=p,
        v=v,
        A=A,
        log_a_cum=log_a_cum,
        bt=bt,
        u=u,
        y=y,
        z=z,
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
    return u, y, z


def prepare_uyz_repr_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    beta: torch.Tensor,
    log_a_cum: torch.Tensor,
    log_mu_cum: torch.Tensor,
    log_ct: torch.Tensor,
    gamma_mask_q: torch.Tensor,
    d_Attn_do_v: torch.Tensor,
    d_decay_s: torch.Tensor,
    A: torch.Tensor,
    bt: torch.Tensor,
    dbt: torch.Tensor,
    d_log_mu_cum: torch.Tensor,
    d_log_a_cum: torch.Tensor,
    du: torch.Tensor,
    dy: torch.Tensor,
    dz: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = gamma_mask_q.shape[-1]
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # H100 can have larger block size
    # if check_shared_mem('hopper', k.device.index):
    #     CONST_TILING = 128
    if check_shared_mem:
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    # CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dv = torch.empty_like(v)

    dk = torch.empty_like(k, dtype=torch.float)
    dp = torch.empty_like(p, dtype=torch.float)

    dlog_a = torch.empty_like(log_a_cum, dtype=torch.float)
    dlog_mu = torch.empty_like(log_mu_cum, dtype=torch.float)
    dbeta = torch.empty_like(log_mu_cum, dtype=torch.float)

    prepare_uyz_repr_bwd_kernel[(NT, B * H)](
        q=q,
        k=k,
        v=v,
        p=p,
        beta=beta,
        log_a_cum=log_a_cum,
        log_m_cum=log_mu_cum,
        gamma_mask_q=gamma_mask_q,
        d_Attn_do_v=d_Attn_do_v,
        d_decay_s=d_decay_s,
        bt=bt,
        log_ct=log_ct,
        A=A,
        du=du,
        dy=dy,
        dz=dz,
        dk=dk,
        dv=dv,
        dp=dp,
        dlog_a=dlog_a,
        dlog_mu=dlog_mu,
        dbeta=dbeta,
        dbt=dbt,
        d_log_mu_cum=d_log_mu_cum,
        d_log_a_cum=d_log_a_cum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )

    return dk, dv, dp, dlog_a, dlog_mu, dbeta
