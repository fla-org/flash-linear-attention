
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp


@triton.jit()
def chunk_update_once(
    b_p,
    b_k,
    b_v,
    b_m,
    b_g_exp_q,
    b_h,
    b_lamb
):
    b_o = tl.dot((tl.dot(b_p.to(b_k.dtype), tl.trans(b_k)) * b_m).to(b_v.dtype), b_v)
    b_o += tl.dot((b_p * b_g_exp_q).to(b_h.dtype), b_h)
    if b_lamb is not None:
        b_o += b_lamb[None, :] * b_p
    return b_o


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_mesa_cg_dim64_kernel(
    dq,
    dq_final,
    k,
    h,
    g,
    beta,
    lamb,
    cu_seqlens,
    chunk_indices,
    T,
    max_CG_iteration: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    # offset calculation
    dq += (bos * H + i_h) * K
    dq_final += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    h += (i_tg * H + i_h).to(tl.int64) * K * K

    g += bos * H + i_h
    beta += bos * H + i_h
    lamb += i_h * K

    p_q = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_h = tl.make_block_ptr(h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))

    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)

    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    p_lamb = tl.make_block_ptr(lamb, (K,), (1,), (0,), (BK,), (0,))
    b_lamb = tl.load(p_lamb, boundary_check=(0,)).to(tl.float32)

    b_m = exp(b_g[:, None] - b_g[None, :]) * b_beta[None, :]
    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), b_m, 0)
    b_g_exp_q = tl.exp(b_g)[:, None]

    b_x = tl.zeros([BT, BK], dtype=tl.float32)
    b_p = tl.zeros([BT, BK], dtype=tl.float32)
    b_r = tl.zeros([BT, BK], dtype=tl.float32)

    b_x += b_q * 0.
    b_r += b_q
    b_p += b_q
    b_delta_old = tl.sum(b_r*b_r, axis=1)
    for _ in range(max_CG_iteration):
        b_o = chunk_update_once(b_p, b_k, b_k, b_m, b_g_exp_q, b_h, b_lamb)
        alpha = b_delta_old / (tl.sum(b_p*b_o, axis=1) + 1e-5)
        b_x += alpha[:, None] * b_p
        b_r = b_r - alpha[:, None] * b_o
        b_delta_new = tl.sum(b_r*b_r, axis=1)
        b_p = b_r + (b_delta_new / (b_delta_old + 1e-5))[:, None] * b_p
        b_delta_old = b_delta_new

    p_q_final = tl.make_block_ptr(dq_final, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_final, b_x.to(p_q_final.dtype.element_ty), boundary_check=(0, 1))


def chunk_mesa_cg_bwd(
    dq: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    g_local_cumsum: torch.Tensor,
    beta: torch.Tensor,
    lamb: torch.Tensor,  # lambda
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    max_CG_iteration: int = 30,
    output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    B, T, H, K = dq.shape
    assert K <= 128, "head dimension must be less than 128"
    assert chunk_size <= 64 or K <= 64, "either chunk size or head dimension must be no greater than 64"
    dq_final = torch.empty_like(dq, dtype=dq.dtype if output_dtype is None else output_dtype)

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    BK = triton.next_power_of_2(K)
    grid = (NT, H*B)

    chunk_fwd_mesa_cg_dim64_kernel[grid](
        dq=dq,
        dq_final=dq_final,
        k=k,
        h=h,
        g=g_local_cumsum,
        beta=beta,
        lamb=lamb,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        max_CG_iteration=max_CG_iteration,
        T=T,
        H=H,
        K=K,
        BT=chunk_size,
        BK=BK,
        num_warps=4,
        num_stages=1
    )
    return dq_final
