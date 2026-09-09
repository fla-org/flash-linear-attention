# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs, check_shared_mem


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'USE_SINK_BIAS': lambda args: args['sink_bias'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([] if check_shared_mem('hopper') else [8])
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'USE_SINK_BIAS'],
    **autotune_cache_kwargs,
)
@triton.jit
def naive_attn_decoding_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    sink_bias,
    scale,
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
    USE_G: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v, i_bh = pid % NV, (pid // NV).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos, eos = tl.load(cu_seqlens + i_b).to(tl.int64), tl.load(cu_seqlens + i_b + 1).to(tl.int64)
    T = eos - bos

    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + i_bh * K + o_d
    p_o = o + i_bh * V + o_v

    b_q = tl.load(p_q, mask=o_d < K, other=0.0)
    b_q = (b_q * scale).to(b_q.dtype)

    b_o = tl.zeros([BV], dtype=tl.float32)

    b_m = tl.full([1], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([1], dtype=tl.float32)

    if USE_G:
        p_g = g_cumsum + bos * HQ + i_hq + (T - 1) * HQ
        b_gq = tl.load(p_g, mask=(T - 1) < T, other=0.0).to(tl.float32)
    else:
        b_gq = None

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32)
    else:
        b_sink_bias = None

    for i_s in range(0, T, BS):
        o_k = (i_s + tl.arange(0, BS)).to(tl.int64)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_k[:, None] * (H*K) + o_d[None, :]
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        # [BK, BS]
        b_k = tl.load(p_k, mask=m_k[:, None] & (o_d[None, :] < K), other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        # [BT, BS]
        b_s = tl.sum(b_q[None, :] * b_k, 1)

        b_s = tl.where(m_k, b_s, float('-inf'))

        if USE_G:
            p_gk = g_cumsum + bos * HQ + i_hq + o_k * HQ
            b_gk = tl.load(p_gk, mask=m_k, other=0.0).to(tl.float32)
            b_s += b_gq - b_gk
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

    if USE_SINK_BIAS:
        # keep the sink-bias merge finite when masking leaves a row with no valid key.
        b_m = tl.where(b_m == float('-inf'), 0., b_m)
        b_acc += exp(b_sink_bias - b_m)
    b_o = b_o / b_acc
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=o_v < V)


def attn_decoding_one_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor = None,
    do_gate_scale: bool = False,
    *,
    sink_bias: torch.Tensor | None = None,
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
            log decay factors of shape `[1, T, HQ]`. Default: `None`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        do_gate_scale (bool):
            Whether to apply gate scale. Default: `False`. If `True`, the attention scale will also be applied
            to the gating bias term in Forgetting Transformer or PaTH-FoX.
        sink_bias (Optional[torch.Tensor]):
            Per-query-head attention-sink bias logits of shape `[HQ]` — one
            learnable scalar per query head, as introduced by GPT-OSS.
            Augments the softmax denominator without contributing to the output.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[1, B, HQ, V]`.
    """
    assert cu_seqlens is not None, "The cu_seqlens must be provided for varlen decoding"
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = len(cu_seqlens) - 1
    HQ = q.shape[2]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5
    if sink_bias is not None:
        assert sink_bias.shape == (HQ,), "sink_bias must have shape [HQ]"

    BK = max(triton.next_power_of_2(K), 16)
    if check_shared_mem('hopper', q.device.index):
        BS = min(64, max(16, triton.next_power_of_2(T)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
    elif check_shared_mem('ampere', q.device.index):
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
    else:
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
    g_cumsum = chunk_global_cumsum(
        g,
        cu_seqlens=cu_seqlens,
        scale=scale if do_gate_scale else None,
        output_dtype=torch.float32,
    ) if g is not None else None
    NV = triton.cdiv(V, BV)
    o = torch.empty(*q.shape[:-1], V, dtype=v.dtype, device=q.device)

    grid = (NV * N * HQ,)
    naive_attn_decoding_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        g_cumsum=g_cumsum,
        sink_bias=sink_bias,
        scale=scale,
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
