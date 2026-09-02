# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices

NUM_WARPS = [1, 2, 4, 8, 16]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN']
)
@triton.jit(do_not_specialize=['T'])
def chunk_mode_rule_cumsum_scalar_fwd_kernel(
        log_alpha,
        log_mu,
        beta,
        log_a_cum,
        log_mu_cum,
        log_ct,
        cu_seqlens,
        chunk_indices,
        T,
        B: tl.constexpr,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_t_cur = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        i_t = i_t_cur
        T = eos - bos
    else:
        bos = i_b * T
        eos = bos + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    base = log_alpha + bos * H + i_h
    b_log_a = tl.load(base + o_t * H, mask=m_t, other=0.0).to(tl.float32)
    b_log_m = tl.load(log_mu + bos * H + i_h + o_t * H, mask=m_t, other=0.0).to(tl.float32)
    b_beta = tl.load(beta + bos * H + i_h + o_t * H, mask=m_t, other=0.0).to(tl.float32)

    eps = 1e-6
    b_log_a_cum = tl.cumsum(b_log_a, axis=0)
    b_log_m_cum = tl.cumsum(b_log_m, axis=0)
    b_log_beta = tl.log(b_beta + eps)
    b_log_c = b_log_beta + tl.cumsum(b_log_m - b_log_a, axis=0)

    neg_inf = float("-inf")
    # logcumsumexp
    b_log_c_matrix = tl.where((o_t[:, None] >= o_t[None, :]) & m_t[None, :], b_log_c[None, :], neg_inf)
    # tl.max with float -inf needs handling; use tl.where for masked
    b_row_max = tl.max(b_log_c_matrix, axis=1)
    b_ct = tl.sum(tl.exp(b_log_c_matrix - b_row_max[:, None]), axis=1)
    b_log_ct = tl.log(b_ct) + b_row_max
    # for padded entries, keep 0
    b_log_ct = tl.where(m_t, b_log_ct, 0.0)
    b_log_a_cum = tl.where(m_t, b_log_a_cum, 0.0)
    b_log_m_cum = tl.where(m_t, b_log_m_cum, 0.0)

    tl.store(log_a_cum + bos * H + i_h + o_t * H, b_log_a_cum, mask=m_t)
    tl.store(log_mu_cum + bos * H + i_h + o_t * H, b_log_m_cum, mask=m_t)
    tl.store(log_ct + bos * H + i_h + o_t * H, b_log_ct, mask=m_t)


def chunk_mode_rule_cumsum_scalar_fwd(
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int,
        cu_seqlens: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H = log_alpha.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    log_a_cum = torch.empty_like(log_alpha, dtype=output_dtype or log_alpha.dtype)
    log_mu_cum = torch.empty_like(log_mu, dtype=output_dtype or log_mu.dtype)
    log_ct = torch.empty_like(log_alpha, dtype=output_dtype or log_alpha.dtype)

    grid = (NT, B * H)
    chunk_mode_rule_cumsum_scalar_fwd_kernel[grid](
        log_alpha=log_alpha,
        log_mu=log_mu,
        beta=beta,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        log_ct=log_ct,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
    )
    return log_a_cum, log_mu_cum, log_ct


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN']
)
@triton.jit(do_not_specialize=['T'])
def chunk_mode_rule_cumsum_scalar_bwd_kernel(
        d_log_a_cum,
        d_log_mu_cum,
        d_log_bar_a_tm1,
        d_log_cum_1_t,
        log_c_cum1t,
        log_c_before,
        dbeta,
        dlog_alpha,
        dlog_mu,
        cu_seqlens,
        chunk_indices,
        T,
        B: tl.constexpr,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_t_cur = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        i_t = i_t_cur
        T = eos - bos
    else:
        bos = i_b * T
        eos = bos + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_dlog_a_cum = tl.load(d_log_a_cum + bos * H + i_h + o_t * H, mask=m_t, other=0.0).to(tl.float32)
    b_dlog_mu_cum = tl.load(d_log_mu_cum + bos * H + i_h + o_t * H, mask=m_t, other=0.0).to(tl.float32)
    b_dlog_bar_a_tm1 = tl.load(d_log_bar_a_tm1 + bos * H + i_h + o_t * H, mask=m_t, other=0.0).to(tl.float32)
    b_dlog_cum_1_t_shift = tl.load(d_log_cum_1_t + bos * H + i_h + o_t * H, mask=m_t, other=0.0).to(tl.float32)
    b_log_cum_1_t = tl.load(log_c_cum1t + bos * H + i_h + o_t * H, mask=m_t, other=0.0).to(tl.float32)
    b_log_c_before = tl.load(log_c_before + bos * H + i_h + o_t * H, mask=m_t, other=0.0).to(tl.float32)

    w = b_dlog_bar_a_tm1 * tl.exp(-b_log_cum_1_t)
    r = tl.cumsum(w, axis=0, reverse=True)

    offs = tl.arange(0, BT)
    mask = (offs > 0) & (offs < BT - 1)
    # m_t already masks padded, combine
    b_dlog_cum_1_t_shift = tl.where(mask & m_t, b_dlog_cum_1_t_shift, 0.0)

    d_log_c_before = tl.exp(b_log_c_before) * r

    b_dlog_a_cum = b_dlog_a_cum + b_dlog_cum_1_t_shift - d_log_c_before
    b_dlog_mu_cum = b_dlog_mu_cum + d_log_c_before
    b_dlog_beta = d_log_c_before

    b_dlog_alpha = tl.cumsum(b_dlog_beta, axis=0, reverse=True)
    b_dlog_mu = tl.cumsum(b_dlog_mu_cum, axis=0, reverse=True)

    tl.store(dlog_alpha + bos * H + i_h + o_t * H, b_dlog_alpha, mask=m_t)
    tl.store(dlog_mu + bos * H + i_h + o_t * H, b_dlog_mu, mask=m_t)
    tl.store(dbeta + bos * H + i_h + o_t * H, b_dlog_beta, mask=m_t)


def chunk_mode_rule_cumsum_scalar_bwd(
        d_log_a_cum: torch.Tensor,
        d_log_mu_cum: torch.Tensor,
        d_log_bar_a_tm1: torch.Tensor,
        d_log_cum_1_t: torch.Tensor,
        log_c_cum1t: torch.Tensor,
        log_c_before: torch.Tensor,
        chunk_size: int,
        cu_seqlens: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = torch.float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H = d_log_a_cum.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dbeta = torch.empty_like(d_log_a_cum, dtype=output_dtype or d_log_a_cum.dtype)
    dlog_alpha = torch.empty_like(d_log_a_cum, dtype=output_dtype or d_log_a_cum.dtype)
    dlog_mu = torch.empty_like(d_log_a_cum, dtype=output_dtype or d_log_a_cum.dtype)

    grid = (NT, B * H)
    chunk_mode_rule_cumsum_scalar_bwd_kernel[grid](
        d_log_a_cum=d_log_a_cum,
        d_log_mu_cum=d_log_mu_cum,
        d_log_bar_a_tm1=d_log_bar_a_tm1,
        d_log_cum_1_t=d_log_cum_1_t,
        dbeta=dbeta,
        dlog_alpha=dlog_alpha,
        dlog_mu=dlog_mu,
        log_c_cum1t=log_c_cum1t,
        log_c_before=log_c_before,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
    )
    return dbeta, dlog_alpha, dlog_mu
