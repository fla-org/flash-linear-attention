# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, Union, Tuple

import torch
from einops import repeat
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention
from fla.ops.utils.pooling import mean_pooling

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

def naive_nsa_sel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int = 64,
    scale: Optional[float] = None,
    cu_seqlens: Union[None, torch.LongTensor, Tuple[torch.LongTensor]] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, TQ, HQ, K]`..
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, TQ, H, S]` if `head_first=False` else `[B, H, TQ, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
        block_size (int):
            Selected block size. Default: 64.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor or Tuple[torch.LongTensor]):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
            When a tuple is provided, it should contain two tensors: `(cu_seqlens_q, cu_seqlens_k)`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, TQ, HQ, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
    q, k, v = map(lambda x: x.float(), (q, k, v))
    B = q.shape[0]

    o = torch.zeros_like(v)
    varlen = True
    if cu_seqlens is None:
        varlen = False
        Tq = Tk = q.shape[1]
        cu_q = torch.cat([
            block_indices.new_tensor(range(0, B * Tq, Tq)), block_indices.new_tensor([B * Tq])
        ])
        cu_k = torch.cat([
            block_indices.new_tensor(range(0, B * Tk, Tk)), block_indices.new_tensor([B * Tk])
        ])
    else:
        if isinstance(cu_seqlens, tuple):
            cu_q, cu_k = cu_seqlens
        else:
            cu_q = cu_k = cu_seqlens

    for i in range(len(cu_q) - 1):
        if not varlen:
            q_b, k_b, v_b, i_b = q[i], k[i], v[i], block_indices[i]
        else:
            Tq = cu_q[i+1] - cu_q[i]
            Tk = cu_k[i+1] - cu_k[i]
            q_b, k_b, v_b, i_b = q[0][cu_q[i]:cu_q[i+1]], k[0][cu_k[i]:cu_k[i+1]], v[0][cu_k[i]:cu_k[i+1]], block_indices[0][cu_q[i]:cu_q[i+1]]

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, S*BS, HQ]
        i_b = i_b.view(Tq, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(Tq):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [S*BS, HQ]
            i_i = i_b[i_q]
            # [S*BS, HQ, -1]
            k_i, v_i = map(lambda x: x.gather(0, i_i.clamp(0, Tk-1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])), (k_b, v_b))
            # [S*BS, HQ]
            attn = torch.einsum('h d, n h d -> n h', q_i, k_i).masked_fill(i_i > i_q, float('-inf')).softmax(0)
            if not varlen:
                o[i, i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)
            else:
                o[0][cu_q[i] + i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)

    return o.to(dtype)

def naive_nsa_cmp(q, k_cmp, v_cmp, block_size, scale, cu_seqlens=None):
    assert cu_seqlens is None, "Not implemented yet"

    @torch.compile
    def cmp_mask(b, h, q_idx, kv_idx):
        return q_idx >= (kv_idx + 1) * block_size - 1

    B, H, TQ, TKV = q.shape[0], k_cmp.shape[1], q.shape[1], k_cmp.shape[1]
    block_mask = create_block_mask(cmp_mask, B, H, TQ, TKV)

    o_cmp, lse_cmp = flex_attention(
        q.transpose(1, 2),
        k_cmp.transpose(1, 2),
        v_cmp.transpose(1, 2),
        block_mask=block_mask,
        enable_gqa=True,
        return_lse=True,
        scale=scale,
    )
    return o_cmp.transpose(1, 2), lse_cmp.transpose(1, 2)


def naive_nsa_topk(
    q: torch.Tensor,               # [B, T_q, Hq, D]
    k_cmp: torch.Tensor,           # [B, T_C, Hkv, D]  (T_C = #compressed blocks)
    block_counts: Union[int, torch.Tensor],  # int or [B, T_q, Hkv]
    block_size: int,
    scale: float,
    cu_seqlens: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
) -> torch.Tensor:
    assert q.ndim == 4 and k_cmp.ndim == 4, "q:[B,Tq,Hq,D], k_cmp:[B,Tc,Hkv,D]"
    B, Tq, Hq, D = q.shape
    B2, Tc, Hkv, D2 = k_cmp.shape
    assert B == B2 and D == D2, "Batch and depth must match between q and k_cmp"
    device = q.device

    # ---- 1) GQA logits: per-Q-head unnormalized scores ----
    assert Hq % Hkv == 0
    G = Hq // Hkv
    qg = q.reshape(B, Tq, Hkv, G, D)                                        # [B,Tq,Hkv,G,D]
    logits = torch.einsum('b t h g d, b s h d -> b t h g s', qg, k_cmp)     # [B,Tq,Hkv,G,Tc]
    logits = logits * float(scale)

    # ---- 2) Block-causal + length masks on UNNORMALIZED logits ----
    # Base causal: allow blocks whose *first* token position <= current query index
    t = torch.arange(Tq, device=device).unsqueeze(1)                         # [Tq,1]
    s = torch.arange(Tc, device=device).unsqueeze(0)                         # [1,Tc]
    block_last_pos = (s + 1) * block_size - 1                                # [1,Tc]
    base_allow = (block_last_pos <= t)                                       # [Tq,Tc]

    # Current block indicator (independent of lengths)
    i_qb = (t // block_size)                                                 # [Tq,1]
    is_current_block = (s == i_qb) | (s == 0) | (s == i_qb - 1)              # [Tq,Tc]

    # Per-sample valid lengths from cumulative seqlens
    def _to_lens(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        assert x.ndim == 1 and x.numel() == B + 1, "cumulative lens must be [B+1]"
        return (x[1:] - x[:-1]).to(torch.long)                               # [B]

    if cu_seqlens is not None:
        q_cu, k_cu = cu_seqlens

        q_len = _to_lens(q_cu)                                               # tokens per sample [B] or None
        k_len = _to_lens(k_cu)                                               # tokens per sample [B] or None
        k_blocks_len = None
        if k_len is not None:
            k_blocks_len = (k_len + (block_size - 1)) // block_size          # ceil_div -> blocks per sample [B]
    else:
        q_len = k_len = k_blocks_len = None

    # Build [B,Tq,Tc] "allow" mask combining causal + per-sample lengths
    allow = base_allow.unsqueeze(0).expand(B, -1, -1)                        # [B,Tq,Tc]
    if q_len is not None:
        q_valid = (torch.arange(Tq, device=device).view(1, Tq, 1) <
                   q_len.view(B, 1, 1))                                      # [B,Tq,1]
        allow = allow & q_valid
    if k_blocks_len is not None:
        k_valid = (torch.arange(Tc, device=device).view(1, 1, Tc) <
                   k_blocks_len.view(B, 1, 1))                                # [B,1,Tc]
        allow = allow & k_valid

    # Apply mask to logits: disallowed blocks -> -inf
    allow_exp = allow[:, :, None, None, :]                                   # [B,Tq,1,1,Tc]
    logits = logits.masked_fill(~allow_exp, float('-inf'))
    allow = allow | is_current_block[None, :, :]

    # ---- 3) Per-Q-head softmax over blocks, then mean across G ----
    probs_q = torch.softmax(logits, dim=-1)                                  # [B,Tq,Hkv,G,Tc]
    probs_q = torch.nan_to_num(probs_q, nan=0.0)                              # rows with no valid blocks -> 0
    scores = probs_q.mean(dim=3)                                             # [B,Tq,Hkv,Tc]

    # ---- 4) Set the current block's normalized score to 1.0 (when allowed) ----
    scores = torch.where(is_current_block[None, :, None, :], 1.0, scores)

    # ---- 5) Top-k selection per [B, Tq, Hkv] ----
    if isinstance(block_counts, int):
        S = int(block_counts)
        assert S >= 0, "block_counts (int) must be >= 0"
        desired_k = torch.full((B, Tq, Hkv), S, dtype=torch.long, device=device)
    elif torch.is_tensor(block_counts):
        assert block_counts.shape == (B, Tq, Hkv)
        desired_k = block_counts.to(device=device, dtype=torch.long)
        S = int(torch.clamp(desired_k, min=0).max().item())
    else:
        raise TypeError("block_counts must be int or torch.Tensor")

    if S == 0:
        return torch.empty(B, Tq, Hkv, 0, dtype=torch.long, device=device)

    _, topi = torch.topk(scores, k=min(S, Tc), dim=-1)                             # [B,Tq,Hkv,S]

    # Validate selections against allow mask; pad with -1 where invalid or beyond quota
    allow_kv = allow.unsqueeze(2).expand(B, Tq, Hkv, Tc)                      # [B,Tq,Hkv,Tc]
    sel_allowed = torch.gather(allow_kv.long(), dim=-1, index=topi).bool()    # [B,Tq,Hkv,S]

    idx = torch.arange(S, device=device).view(1, 1, 1, S)
    within_quota = (idx < desired_k.unsqueeze(-1))[:, :, :, :Tc]                              # [B,Tq,Hkv,S]

    keep = sel_allowed & within_quota
    out = torch.full_like(topi, fill_value=-1)                                # pad with -1
    out = torch.where(keep, topi, out)

    if S > Tc:
        out = torch.cat((out, torch.full((B, Tq, Hkv, S - Tc), -1,
                                         device=device, dtype=topi.dtype)), dim=-1)

    return out


def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: Optional[torch.Tensor] = None,
    g_slc: Optional[torch.Tensor] = None,
    g_swa: Optional[torch.Tensor] = None,
    block_indices: Optional[torch.LongTensor] = None,
    block_counts: Union[torch.LongTensor, int] = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, TQ, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g_cmp (torch.Tensor):
            Gate score for compressed attention of shape `[B, TQ, HQ]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, TQ, HQ]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, TQ, HQ]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, TQ, H, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
            If `g_cmp` is provided, the passed `block_indices` will be ignored.
        block_counts (Optional[Union[torch.LongTensor, int]]):
            Number of selected blocks for each query.
            If a tensor is provided, with shape `[B, TQ, H]`,
            each query can select the same number of blocks.
            If not provided, it will default to 16.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
    """
    assert block_counts is not None, "block counts must be provided for selection"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    assert q.shape[2] % (k.shape[2] * 16) == 0, "Group size must be a multiple of 16 in NSA"

    k_cmp, v_cmp = mean_pooling(k, block_size, cu_seqlens), mean_pooling(v, block_size, cu_seqlens)
    o_cmp, lse_cmp = None, None
    if g_cmp is not None:
        o_cmp, lse_cmp = naive_nsa_cmp(
            q=q,
            k_cmp=k_cmp,
            v_cmp=v_cmp,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens
        )
        if block_indices is not None:
            warnings.warn("`block_indices` will be ignored when `g_cmp` is provided")
        block_indices = naive_nsa_topk(
            q=q,
            k_cmp=k_cmp,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens
        )
    o = o_slc = naive_nsa_sel(q, k, v, block_indices, block_size, scale, cu_seqlens)
    if g_slc is not None:
        o = o_slc * g_slc.unsqueeze(-1)
    if o_cmp is not None:
        o = torch.addcmul(o, o_cmp, g_cmp.unsqueeze(-1))
    if window_size > 0:
        if cu_seqlens is not None:
            max_seqlen = q.shape[1]
            o_swa = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(window_size-1, 0)
            ).unsqueeze(0)
        else:
            o_swa = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(window_size-1, 0)
            )
        o = torch.addcmul(o, o_swa, g_swa.unsqueeze(-1))
    return o
