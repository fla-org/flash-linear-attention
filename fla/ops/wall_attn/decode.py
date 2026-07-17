# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Wall attention, contributed by Tilde Research (Timor Averbuch, Dhruv Pai).
# Heavily modified from fla/ops/gated_oja_rule/chunk.py.
"""Wall single-step decode kernel and pre-rescaled KV-cache builder."""

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2, log2
from fla.utils import autotune_cache_kwargs, check_shared_mem

WALL_DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({'BT': BT}, num_warps=nw, num_stages=ns)
    for BT in (16, 32)
    for nw in (2, 4, 8)
    for ns in (2, 3)
]


@triton.autotune(
    configs=WALL_DECODE_AUTOTUNE_CONFIGS,
    key=['KV_CHUNKS_BUCKET', 'K', 'V', 'HQ', 'H', 'C'],
    **autotune_cache_kwargs,
)
@triton.heuristics({
    'USE_SINK_BIAS': lambda args: args['sink_bias'] is not None,
    'USE_SCALAR_G': lambda args: args['g_scalar_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T_kv', 'NC', 'KV_CHUNKS_BUCKET'])
def parallel_wall_attn_decode_kernel(
    q,                # [B, T_q, HQ, K]
    k_tilde,          # [B, T_kv, HQ, K]  pre-rescaled keys (per-Q-head)
    v,                # [B, T_kv, H,  V]
    o,                # [B, T_q, HQ, V]
    p_curr,           # [B, T_q, HQ, K]
    r_cache,          # [B, NC,  HQ, K]  per-chunk anchors
    g_scalar_cumsum,  # [B, T_kv, HQ] or None  (cache-side scalar cumsum; query-side at tail)
    sink_bias,        # [HQ] or None
    lse,              # [B, T_q, HQ]
    scale,
    T_q,
    T_kv,
    NC,
    KV_CHUNKS_BUCKET,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    C: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    RCP_LN2: tl.constexpr = 1.4426950216

    bos_q = (i_b * T_q).to(tl.int64)
    bos_kv = (i_b * T_kv).to(tl.int64)
    bos_nc = (i_b * NC).to(tl.int64)

    p_q = tl.make_block_ptr(
        q + (bos_q * HQ + i_hq) * K, (T_q, K), (HQ * K, 1),
        (i_t * BT, 0), (BT, BK), (1, 0),
    )
    p_pq = tl.make_block_ptr(
        p_curr + (bos_q * HQ + i_hq) * K, (T_q, K), (HQ * K, 1),
        (i_t * BT, 0), (BT, BK), (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos_q * HQ + i_hq) * V, (T_q, V), (HQ * V, 1),
        (i_t * BT, i_v * BV), (BT, BV), (1, 0),
    )
    p_lse = tl.make_block_ptr(
        lse + bos_q * HQ + i_hq, (T_q,), (HQ,), (i_t * BT,), (BT,), (0,),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)

    if USE_SCALAR_G:
        o_q_global = (T_kv - T_q) + i_t * BT + tl.arange(0, BT)
        m_q = o_q_global < T_kv
        b_cq = tl.load(
            g_scalar_cumsum + (bos_kv + o_q_global) * HQ + i_hq,
            mask=m_q, other=0,
        ).to(tl.float32)

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32) * RCP_LN2

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # Iterate over the KV cache one chunk at a time: BS == C
    for i_s in range(0, T_kv, BS):
        i_c = i_s // C
        # Load anchor
        p_r = tl.make_block_ptr(
            r_cache + (bos_nc * HQ + i_hq) * K, (NC, K), (HQ * K, 1),
            (i_c, 0), (1, BK), (1, 0),
        )
        b_R = tl.load(p_r, boundary_check=(0, 1)).to(tl.float32)
        b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

        p_kt = tl.make_block_ptr(
            k_tilde + (bos_kv * HQ + i_hq) * K, (K, T_kv), (1, HQ * K),
            (0, i_s), (BK, BS), (0, 1),
        )
        p_v = tl.make_block_ptr(
            v + (bos_kv * H + i_h) * V, (T_kv, V), (H * V, 1),
            (i_s, i_v * BV), (BS, BV), (1, 0),
        )
        b_kt = tl.load(p_kt, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q_til, b_kt) * scale * RCP_LN2

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T_kv
        if USE_SCALAR_G:
            b_ck = tl.load(
                g_scalar_cumsum + (bos_kv + o_k) * HQ + i_hq,
                mask=m_k, other=0,
            ).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]

        b_s = tl.where(m_k[None, :], b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float('-inf'), 0., b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)

    if USE_SINK_BIAS:
        b_m = tl.where(b_m == float('-inf'), 0., b_m)
        b_acc += exp2(b_sink_bias - b_m)

    b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


def parallel_wall_attn_decode(
    q: torch.Tensor,
    v: torch.Tensor,
    p_curr: torch.Tensor,
    k_tilde: torch.Tensor,
    r_cache: torch.Tensor,
    sink_bias: torch.Tensor | None,
    scale: float,
    cache_chunk_size: int,
    g_scalar_cumsum: torch.Tensor | None = None,
):
    r"""Wall decode with a pre-rescaled KV cache.

    Use this directly from modeling code; it does NOT share a signature with
    `parallel_wall_attn_fwd`. The caller is responsible for picking which to
    call (e.g. training vs. cached generation).

    Shapes:
        q        : ``[B, T_q,  HQ, K]``  current queries
        v        : ``[B, T_kv, H,  V]``  cached values (standard layout)
        p_curr   : ``[B, T_q,  HQ, K]``  per-channel prefix at the current rows
        k_tilde  : ``[B, T_kv, HQ, K]``  pre-rescaled keys (see `build_wall_kv_cache`)
        r_cache  : ``[B, NC,   HQ, K]``  per-chunk anchors, ``NC = ceil(T_kv / C)``
        g_scalar_cumsum : ``[B, T_kv, HQ]`` or ``None``  optional FoX-style scalar gate

    Returns ``(o, lse)`` with ``o.shape == [B, T_q, HQ, V]``.
    """
    C = cache_chunk_size
    B, T_q, HQ, K = q.shape
    _, T_kv, H, V = v.shape
    NC = r_cache.shape[1]
    G = HQ // H
    # the bucket selects an autotune config; exact T_kv remains the runtime loop bound
    KV_CHUNKS_BUCKET = triton.next_power_of_2(max(1, triton.cdiv(T_kv, C)))

    if k_tilde.shape != (B, T_kv, HQ, K):
        raise ValueError(f"k_tilde shape {k_tilde.shape} != expected {(B, T_kv, HQ, K)}")
    if r_cache.shape != (B, NC, HQ, K):
        raise ValueError(f"r_cache shape {r_cache.shape} != expected {(B, NC, HQ, K)}")
    if p_curr.shape != (B, T_q, HQ, K):
        raise ValueError(f"p_curr shape {p_curr.shape} != expected {(B, T_q, HQ, K)}")
    if T_kv > NC * C:
        raise ValueError(f"NC*C ({NC * C}) < T_kv ({T_kv}); cache anchors do not cover cache")
    if HQ % H != 0:
        raise ValueError(f"HQ ({HQ}) must be divisible by H ({H})")

    for t in (q, v, p_curr, k_tilde, r_cache):
        if t.stride(-1) != 1:
            raise ValueError("decode tensors must be contiguous in the last (K or V) dim")
    if g_scalar_cumsum is not None and g_scalar_cumsum.stride(-1) != 1:
        raise ValueError("g_scalar_cumsum must be contiguous in the last dim")

    if check_shared_mem('hopper', q.device.index):
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
    elif check_shared_mem('ampere', q.device.index):
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
    else:
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
    NV = triton.cdiv(V, BV)
    assert triton.cdiv(K, BK) == 1, "decode kernel requires K <= 256"

    o = torch.empty(B, T_q, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T_q, HQ, dtype=torch.float, device=q.device)

    def grid(meta):
        return (NV, triton.cdiv(T_q, meta['BT']), B * HQ)
    parallel_wall_attn_decode_kernel[grid](
        q=q,
        k_tilde=k_tilde,
        v=v,
        o=o,
        p_curr=p_curr,
        r_cache=r_cache,
        g_scalar_cumsum=g_scalar_cumsum,
        sink_bias=sink_bias,
        lse=lse,
        scale=scale,
        T_q=T_q,
        T_kv=T_kv,
        NC=NC,
        KV_CHUNKS_BUCKET=KV_CHUNKS_BUCKET,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BS=C,
        BK=BK,
        BV=BV,
        C=C,
    )
    return o, lse


def build_wall_kv_cache(
    k: torch.Tensor,         # [B, T, H, K]
    g_cumsum: torch.Tensor,  # [B, T, HQ, K]   per-channel prefix P = cumsum(log_2 g)
    chunk_size: int,
    *,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build a pre-rescaled Wall KV cache.

    Returns ``(k_tilde, r_cache)`` where, for each chunk ``c`` of size ``C``:
        ``R_c          = P[chunk_start_c]``                     ``[B, HQ, K]``
        ``k_tilde[j]   = k[j] * exp2(R_{c(j)} - P[j])``         ``[B, T, HQ, K]``
        ``r_cache[c]   = R_c``                                  ``[B, NC, HQ, K]``

    ``k_tilde`` is per-Q-head (carries the gate). Values are unchanged and
    should be cached as-is. ``g_cumsum`` is the same prefix tensor passed to
    the training forward.
    """
    if k.dim() != 4:
        raise ValueError("k must be [B, T, H, K]")
    if g_cumsum.dim() != 4:
        raise ValueError("g_cumsum must be [B, T, HQ, K]")
    B, T, H, K = k.shape
    _, _, HQ, _ = g_cumsum.shape
    if g_cumsum.shape[:2] != (B, T) or g_cumsum.shape[-1] != K:
        raise ValueError(f"g_cumsum shape {g_cumsum.shape} incompatible with k {k.shape}")
    if HQ % H != 0:
        raise ValueError(f"HQ ({HQ}) must be divisible by H ({H})")
    G = HQ // H
    NC = triton.cdiv(T, chunk_size)
    out_dtype = out_dtype if out_dtype is not None else k.dtype

    # Per-Q-head broadcast of k along the HQ axis (GQA: each KV head feeds G query heads).
    k_q = k.repeat_interleave(G, dim=2)  # [B, T, HQ, K]

    # Per-chunk reference R_c = P[chunk_start, :] for each batch / head.
    # Pad T up to NC*chunk_size with the last row (rescaler will be exp2(0)=1 for tail).
    chunk_starts = torch.arange(NC, device=k.device) * chunk_size
    chunk_starts = chunk_starts.clamp(max=T - 1)
    r_cache = g_cumsum[:, chunk_starts, :, :].contiguous()  # [B, NC, HQ, K]

    # Per-token R(j) = R_{c(j)}
    token_chunk = torch.arange(T, device=k.device) // chunk_size  # [T]
    r_per_token = r_cache[:, token_chunk, :, :]  # [B, T, HQ, K]

    k_tilde = (k_q.to(torch.float32) * torch.exp2(r_per_token.to(torch.float32) - g_cumsum.to(torch.float32)))

    return k_tilde.to(out_dtype).contiguous(), r_cache
