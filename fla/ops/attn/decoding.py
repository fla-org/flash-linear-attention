# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.op import exp
from fla.utils import check_shared_mem


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([] if check_shared_mem('hopper') else [8])
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'G', 'K', 'V', 'BK', 'BV', 'USE_G'],
)
@triton.jit
def naive_attn_decoding_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    scale,
    gate_scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr
):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos, eos = tl.load(cu_seqlens + i_b).to(tl.int32), tl.load(cu_seqlens + i_b + 1).to(tl.int32)
    T = eos - bos

    p_q = tl.make_block_ptr(q + i_bh * K, (K,), (1, ), (0, ), (BK,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * V, (V,), (1, ), (0, ), (BV,), (0,))

    b_q = tl.load(p_q, boundary_check=(0,))
    b_q = (b_q * scale).to(b_q.dtype)

    b_o = tl.zeros([BV, ], dtype=tl.float32)

    b_m = tl.full([1,], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([1,], dtype=tl.float32)

    if USE_G:
        p_g = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (T-1,), (1,), (0,))
        b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None

    for i_s in range(0, T, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.sum(b_q[None, :] * b_k, 1)

        mask = i_s + tl.arange(0, BS) < T
        b_s = tl.where(mask, b_s, float('-inf'))

        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += (b_gq - b_gk) * gate_scale
        # [BT, BS]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s)), b_m
        b_r = exp(b_mp - b_m)
        # [BT, BS]
        b_p = exp(b_s - b_m)

        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 0)
        # [BT, BV]
        b_o = b_o * b_r + tl.sum(b_p[:, None] * b_v, 0)
        b_mp = b_m
    b_o = b_o / b_acc
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, ))


def attn_decoding_one_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: torch.LongTensor = None,
    do_gate_scale: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            query of shape `[1, B, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[1, T, H, K]`.
            GQA will be applied if HQ is divisible by H. T is the cumulative length for all batch.
        v (torch.Tensor):
            values of shape `[1, T, H, V]`.
        g (Optional[torch.Tensor]):
            log decay factors of shape `[1, T, H]`. Default: `None`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        do_gate_scale (bool):
            Whether to apply gate scale. Default: `False`. If `True`, the attention scale will also be applied
            to the gating bias term in Forgetting Transformer or PaTH-FoX.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, 1, HQ, V]`.
    """
    assert cu_seqlens is not None, "The cu_seqlens must be provided for varlen decoding"
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = len(cu_seqlens) - 1
    HQ = q.shape[2]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5

    BK = triton.next_power_of_2(K)
    if check_shared_mem('hopper', q.device.index):
        BS = min(64, max(16, triton.next_power_of_2(T)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
    elif check_shared_mem('ampere', q.device.index):
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
    else:
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
    g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, output_dtype=torch.float32) if g is not None else None
    NV = triton.cdiv(V, BV)
    o = torch.empty(*q.shape[:-1], V, dtype=v.dtype, device=q.device)
    gate_scale = 1.0 if not do_gate_scale else scale

    grid = (NV, N * HQ)
    naive_attn_decoding_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        g_cumsum=g_cumsum,
        scale=scale,
        gate_scale=gate_scale,
        cu_seqlens=cu_seqlens,
        B=B,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    return o
