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
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
# num_warps is fixed so that dense and varlen calls sum in the same order and give identical results
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=num_stages)
        for num_stages in [1, 2, 3]
    ],
    key=['H', 'K', 'BT', 'NUM_ITER', 'LOAD_FRO'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gka_solve_fwd_kernel(
    q,
    k,
    h,
    gk,
    x,
    fro,
    cu_seqlens,
    chunk_indices,
    ridge_ratio,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NUM_ITER: tl.constexpr,
    LOAD_FRO: tl.constexpr,
    USE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    x += (bos * H + i_h) * K
    fro += bos * H + i_h
    # `h` holds `H_{[c]}`, the state at the start of each chunk, as written by `chunk_fwd_h`
    h += (i_tg * H + i_h) * K*K

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]

    # [BT, BK]
    b_q = tl.load(q + o_t[:, None] * (H*K) + o_k[None, :], mask=m_tk, other=0.)
    b_k = tl.load(k + o_t[:, None] * (H*K) + o_k[None, :], mask=m_tk, other=0.)
    # [BK, BK]
    b_h = tl.load(h + o_k[:, None] * K + o_k[None, :], mask=m_k[:, None] & m_k[None, :], other=0.)

    m_s = (o_t[:, None] >= o_t[None, :]) & m_t[:, None] & m_t[None, :]
    if USE_GK:
        b_gk = tl.load(gk + (bos * H + i_h) + o_t * H, mask=m_t, other=0.).to(tl.float32)
        # [BT, BT] intra-chunk decay, and [BT] decay of the chunk-start state
        b_m = tl.where(m_s, exp2(b_gk[:, None] - b_gk[None, :]), 0).to(b_q.dtype)
        b_gh = exp2(b_gk)
    else:
        b_m = m_s.to(b_q.dtype)
        b_gh = tl.full([BT], 1., dtype=tl.float32)

    if LOAD_FRO:
        b_fro = tl.load(fro + o_t * H, mask=m_t, other=1.)
    else:
        # ||H_t||_F^2 = ||sum_j m_tj k_j k_j^T||^2 + gh_t^2 ||h||^2 + 2 gh_t sum_j m_tj k_j^T h k_j
        b_kk = tl.dot(b_k, tl.trans(b_k))
        b_fro = tl.sum(tl.dot(b_m, (b_kk * b_kk).to(b_q.dtype)) * b_m, axis=1)
        b_fro += b_gh * (b_gh * tl.sum(b_h * b_h))
        b_khk = tl.sum(b_k * tl.dot(b_k.to(b_h.dtype), b_h), axis=1)
        b_fro = tl.sqrt(b_fro + 2 * b_gh * tl.sum(b_m * b_khk[None, :], axis=1))
        # rows past the end of the sequence are never stored; keep their step size finite
        b_fro = tl.where(m_t, b_fro, 1.)

    # Chebyshev iteration on (H_t + lamb_t I) x = q_t; the matrix's spectrum lies in [lamb_t, lamb_t + fro_t]
    b_lamb = (ridge_ratio * b_fro)[:, None]
    b_step = 2 / (2 * b_lamb + b_fro[:, None])
    b_rho = b_fro[:, None] / (2 * b_lamb + b_fro[:, None]) / 2.
    b_rho = b_rho * b_rho

    b_x_prev = tl.zeros([BT, BK], dtype=tl.float32)
    b_x = b_step * b_q.to(tl.float32)
    b_w = tl.full([BT, 1], 2., dtype=tl.float32)
    for _ in range(NUM_ITER):
        b_w = 1. / (1. - b_rho * b_w)
        b_r = (
            tl.dot((b_x * b_gh[:, None]).to(b_h.dtype), b_h)
            + tl.dot((tl.dot(b_x.to(b_k.dtype), tl.trans(b_k)) * b_m).to(b_k.dtype), b_k)
            + b_lamb * b_x
            - b_q
        )
        b_d = b_step * b_w * b_r + (b_w - 1) * b_x_prev
        b_x_prev = b_x
        b_x = b_w * b_x - b_d

    tl.store(
        x + o_t[:, None] * (H*K) + o_k[None, :],
        b_x.to(x.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_tk,
    )
    if not LOAD_FRO:
        tl.store(fro + o_t * H, b_fro.to(fro.dtype.element_ty), mask=m_t)


def chunk_gka_solve_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    gk: torch.Tensor | None = None,
    fro: torch.Tensor | None = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Solves `(H_t + ridge_ratio * ||H_t||_F * I) x_t = q_t` for every token with `num_iter` Chebyshev iterations.

    Args:
        q (torch.Tensor):
            Right-hand sides of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        h (torch.Tensor):
            `H_{[c]}`, the state at the start of each chunk, of shape `[B, NT, H, K, K]`, where `NT` is the number of
            chunks per sequence, or the total over all sequences with `cu_seqlens`.
            It is the output of `chunk_fwd_h(k=k, v=k, g=gk, ...)`.
        gk (torch.Tensor, Optional):
            Chunk-local cumulative sum, in log2 space, of the log-decay `gk_t` in `H_t = exp(gk_t) H_{t-1} + k_t k_t^T`,
            of shape `[B, T, H]`, as returned by `chunk_local_cumsum(gk, chunk_size, scale=RCP_LN2)`. Default: `None`.
        fro (torch.Tensor, Optional):
            Precomputed `||H_t||_F` of shape `[B, T, H]` in `float32`.
            If given, it is reused instead of recomputed, as in the backward pass. Default: `None`.
        ridge_ratio (float, Optional):
            Sets the ridge strength `lamb_t = ridge_ratio * ||H_t||_F`. Default: `0.02`.
        num_iter (int, Optional):
            Number of Chebyshev iterations. Default: `30`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs. Default: `None`.
        chunk_size (int, Optional):
            Chunk size; must match the one used to compute `h`. Default: `64`.
        chunk_indices (torch.LongTensor, Optional):
            Precomputed `prepare_chunk_indices(cu_seqlens, chunk_size)`. Default: `None`.

    Returns:
        A tuple `(x, fro)` where `x` has the shape and dtype of `q` and `fro` is `||H_t||_F` of shape
        `[B, T, H]` in `float32`.
    """
    B, T, H, K = q.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = max(triton.next_power_of_2(K), 16)

    load_fro = fro is not None
    if fro is None:
        fro = q.new_empty(B, T, H, dtype=torch.float)
    x = torch.empty_like(q)
    grid = (NT, B * H)
    chunk_gka_solve_fwd_kernel[grid](
        q=q,
        k=k,
        h=h,
        gk=gk,
        x=x,
        fro=fro,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        ridge_ratio=ridge_ratio,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
        NUM_ITER=num_iter,
        LOAD_FRO=load_fro,
    )
    return x, fro
