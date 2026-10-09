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
def chunk_momentum_delta_rule_cumsum_scalar_fwd_kernel(
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset
    offs = tl.arange(0, BT)
    pos = i_t * BT + offs
    m_pos = pos < T

    base = bos * H + i_h
    p_log_a = log_alpha + base
    p_log_m = log_mu + base
    p_beta = beta + base
    p_log_a_cum = log_a_cum + base
    p_log_mu_cum = log_mu_cum + base
    p_log_ct = log_ct + base

    # [BT]
    b_log_a = tl.load(p_log_a + pos * H, mask=m_pos, other=0.0).to(tl.float32)
    b_log_m = tl.load(p_log_m + pos * H, mask=m_pos, other=0.0).to(tl.float32)
    b_beta = tl.load(p_beta + pos * H, mask=m_pos, other=0.0).to(tl.float32)

    eps = tl.zeros([1], dtype=tl.float32) + 1e-6

    b_log_a_cum = tl.cumsum(b_log_a, axis=0)
    b_log_m_cum = tl.cumsum(b_log_m, axis=0)
    b_log_beta = tl.log(b_beta + eps)

    b_log_c = b_log_beta + tl.cumsum(b_log_m - b_log_a, axis=0)

    # Safe logcumsumexp with O(Chunk_length^2)
    neg_inf = tl.zeros([1], dtype=tl.float32) - float("inf")
    o_t = i_t * BT + offs
    m_A = (o_t[:, None] >= o_t[None, :])
    b_log_c_matrix = tl.where(m_A, b_log_c[None, :], neg_inf)
    b_row_max = tl.max(b_log_c_matrix, axis=1)
    b_ct = tl.sum(tl.exp(b_log_c_matrix - b_row_max[:, None]), axis=1)
    b_log_ct = tl.log(b_ct) + b_row_max

    tl.store(p_log_a_cum + pos * H, b_log_a_cum, mask=m_pos)
    tl.store(p_log_mu_cum + pos * H, b_log_m_cum, mask=m_pos)
    tl.store(p_log_ct + pos * H, b_log_ct, mask=m_pos)


def chunk_momentum_delta_rule_cumsum_scalar_fwd(
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
    chunk_momentum_delta_rule_cumsum_scalar_fwd_kernel[grid](
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
