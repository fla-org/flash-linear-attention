# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp


@triton.jit
def mesa_net_decoding_one_step_kernel(
    q,
    k,
    v,
    g,
    lamb,
    beta,
    prev_h_kk,
    prev_h_kv,
    curr_h_kk,
    curr_h_kv,
    o,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CG_STEP: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_h = i_nh % H

    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)

    p_q = q + i_nh * K + o_k
    p_k = k + i_nh * K + o_k
    p_v = v + i_nh * V + o_v
    p_beta = beta + i_nh
    p_g = g + i_nh
    p_lamb = lamb + i_h * K + o_k

    b_g = exp(tl.load(p_g).to(tl.float32))
    b_beta = tl.load(p_beta).to(tl.float32)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_kk = mask_k[:, None] & mask_k[None, :]
    mask_kv = mask_k[:, None] & mask_v[None, :]

    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    b_lamb = tl.load(p_lamb, mask=mask_k, other=0).to(tl.float32)

    p_hkk_prev = prev_h_kk + i_nh * K * K + o_k[:, None] * K + o_k[None, :]
    b_h_kk = tl.load(p_hkk_prev, mask=mask_kk, other=0).to(tl.float32)

    b_h_kk = b_h_kk * b_g + (b_k * b_beta)[:, None] * b_k[None, :]

    p_hkk_curr = curr_h_kk + i_nh * K * K + o_k[:, None] * K + o_k[None, :]
    tl.store(p_hkk_curr, b_h_kk.to(p_hkk_curr.dtype.element_ty), mask=mask_kk)

    p_hkv_prev = prev_h_kv + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    b_h_kv = tl.load(p_hkv_prev, mask=mask_kv, other=0).to(tl.float32)
    b_h_kv = b_h_kv * b_g + (b_k * b_beta)[:, None] * b_v[None, :]
    p_hkv_curr = curr_h_kv + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p_hkv_curr, b_h_kv.to(p_hkv_curr.dtype.element_ty), mask=mask_kv)

    diag_mask = tl.arange(0, BK)[:, None] == tl.arange(0, BK)[None, :]
    diag_mask = diag_mask & mask_kk
    b_h_kk_diag = tl.sum(tl.where(diag_mask, b_h_kk, 0.0), axis=1)

    b_x = b_q / (b_h_kk_diag + b_lamb + 1e-5)
    b_Hx = tl.sum(b_h_kk * b_x[:, None], axis=0)
    b_r = b_q - b_Hx - b_lamb * b_x
    b_p = tl.zeros([BK,], dtype=tl.float32)
    b_p += b_r
    delta_old = tl.sum(b_r * b_r)

    for i_iter in range(MAX_CG_STEP):
        b_Ap = tl.sum(b_h_kk * b_p[:, None], axis=0) + b_lamb * b_p
        pAp = tl.sum(b_p * b_Ap)
        alpha = delta_old / (pAp + 1e-5)
        b_x = b_x + alpha * b_p
        b_r = b_r - alpha * b_Ap
        delta_new = tl.sum(b_r * b_r)
        beta_cg = delta_new / (delta_old + 1e-5)
        b_p = b_r + beta_cg * b_p
        delta_old = delta_new
    b_o = tl.sum(b_h_kv * b_x[:, None], axis=0)
    p_o = o + i_nh * V + o_v
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)


def mesa_net_decoding_one_step(q, k, v, g, lamb, beta, prev_h_kk, prev_h_kv, max_CG_iteration=30):
    """
    Triton implementation of Mesa Net CG one step

    Args:
        q: Query tensor [B, H, K]
        k: Key tensor [B, H, K]
        v: Value tensor [B, H, V]
        g: Gate tensor [B, H]
        lamb: Lambda tensor [H, K]
        beta: Beta tensor [B, H]
        prev_h_kk: Previous hidden state KK [B, H, K, K]
        prev_h_kv: Previous hidden state KV [B, H, K, V]
        max_CG_iteration: Maximum CG iterations

    Returns:
        o: Output tensor [B, H, V]
        h_kk_new: Updated hidden state KK [B, H, K, K]
        h_kv_new: Updated hidden state KV [B, H, K, V]
    """
    B, H, K = q.shape
    _, _, V = v.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    lamb = lamb.contiguous()
    beta = beta.contiguous()
    prev_h_kk = prev_h_kk.contiguous()
    prev_h_kv = prev_h_kv.contiguous()
    o = torch.empty((B, H, V), dtype=q.dtype, device=q.device)
    curr_h_kk = torch.empty_like(prev_h_kk)
    curr_h_kv = torch.empty_like(prev_h_kv)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    assert BK <= 128 and BV <= 128, "BK and BV must be less than or equal to 128"

    grid = (B * H,)
    mesa_net_decoding_one_step_kernel[grid](
        q=q, k=k, v=v, g=g, lamb=lamb, beta=beta,
        prev_h_kk=prev_h_kk, prev_h_kv=prev_h_kv,
        curr_h_kk=curr_h_kk, curr_h_kv=curr_h_kv,
        o=o,
        B=B, H=H, K=K, V=V,
        BK=BK, BV=BV,
        MAX_CG_STEP=max_CG_iteration,
        num_warps=4 if BK <= 64 else 8
    )
    return o, curr_h_kk, curr_h_kv
