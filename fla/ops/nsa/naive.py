# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional

import torch
from einops import repeat


def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int = 64,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`..
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
        block_size (int):
            Selected block size. Default: 64.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
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

    o = torch.zeros_like(v)
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat([
            block_indices.new_tensor(range(0, B*T, T)), block_indices.new_tensor([B*T])
        ])

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, i_b = q[i], k[i], v[i], block_indices[i]
        else:
            T = cu_seqlens[i+1] - cu_seqlens[i]
            q_b, k_b, v_b, i_b = map(lambda x: x[0][cu_seqlens[i]:cu_seqlens[i+1]], (q, k, v, block_indices))

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, S*BS, HQ]
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [S*BS, HQ]
            i_i = i_b[i_q]
            # [S*BS, HQ, -1]
            k_i, v_i = map(lambda x: x.gather(0, i_i.clamp(0, T-1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])), (k_b, v_b))
            # [S*BS, HQ]
            attn = torch.einsum('h d, n h d -> n h', q_i, k_i).masked_fill(i_i > i_q, float('-inf')).softmax(0)
            if not varlen:
                o[i, i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)
            else:
                o[0][cu_seqlens[i]+i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)

    return o.to(dtype)
