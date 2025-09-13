# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import get_unpad_data, pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.deltaformer import delta_pre_attn
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_func = None

logger = logging.get_logger(__name__)


class DeltaFormerAttention(nn.Module):

    r"""
    The layer implementation for DeltaFormer,
    [Understanding Transformer from the Perspective of Associative Memory]
    (https://arxiv.org/pdf/2505.19488).

    Notes
        - Pre-attention is implemented with Triton kernels in `fla.ops.deltaformer` and is tuned
          for typical head dimensions (e.g., 64/128). It currently supports fixed-length inputs.
        - For variable-length inputs (padding masks), the pre-attention falls back to using the
          fixed-length path, while the second stage (softmax attention over U) uses FlashAttention's
          varlen path when an attention mask is provided.
        - K/V grouping (GQA) is supported via `num_kv_heads` by repeating K/V groups.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        num_heads (int, Optional):
            The number of attention heads. Default: 32.
        num_kv_heads (int, Optional):
            The number of key/value heads for grouped-query attention. If None, equals `num_heads`.
            Default: None.
        qkv_bias (bool, Optional):
            Whether to use bias for Q/K/V projections. Default: False.
        qk_norm (bool, Optional):
            Whether to apply per-head RMSNorm to Q and K before attention. Default: False.
        layer_idx (int, Optional):
            The index of the layer (used for cache compatibility). Default: None.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        layer_idx: int | None = None,
        rope_theta: float = 10000.,
        max_position_embeddings: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.layer_idx = layer_idx
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.b_proj = nn.Linear(self.hidden_size, self.num_kv_heads, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attentions = None
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        beta = rearrange(self.b_proj(hidden_states), 'b t h -> b h t')

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q_len + seqlen_offset
            if attention_mask is not None:
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q_len + max(seqlen_offset)
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        cu_seqlens_rope = kwargs.get('cu_seqlens', None)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens_rope)

        cache_has_content = past_key_values is not None and past_key_values.get_seq_length(self.layer_idx) > 0

        if not cache_has_content or q_len > 1:
            # Prefill: compute U for current block
            if attention_mask is not None:
                _, cu_seqlens_k, _ = get_unpad_data(attention_mask)
                u = delta_pre_attn(
                    rearrange(k, 'b t h d -> b h t d'),  # KK similarity: use k as query
                    rearrange(k, 'b t h d -> b h t d'),
                    rearrange(v, 'b t h d -> b h t d'),
                    beta,
                    cu_seqlens=cu_seqlens_k,
                )
            else:
                if cu_seqlens_rope is not None:
                    lens = (cu_seqlens_rope[1:] - cu_seqlens_rope[:-1]).tolist()
                    assert sum(lens) == q_len and q.shape[0] == 1, "cu_seqlens must cover the flattened sequence"
                    u_slices = []
                    start = 0
                    for L in lens:
                        end = start + int(L)
                        k_b = k[:, start:end, :, :]
                        v_b = v[:, start:end, :, :]
                        beta_b = beta[:, :, start:end]
                        u_b = delta_pre_attn(
                            rearrange(k_b, 'b t h d -> b h t d'),
                            rearrange(k_b, 'b t h d -> b h t d'),
                            rearrange(v_b, 'b t h d -> b h t d'),
                            beta_b,
                        )
                        u_slices.append(rearrange(u_b, 'b h t d -> b t h d'))
                        start = end
                    u = torch.cat(u_slices, dim=1)
                else:
                    u = delta_pre_attn(
                        rearrange(k, 'b t h d -> b h t d'),  # KK similarity: use k as query
                        rearrange(k, 'b t h d -> b h t d'),
                        rearrange(v, 'b t h d -> b h t d'),
                        beta,
                    )
            # If u is [B,H,T,D], rearrange; otherwise already [B,T,H,D]
            if u.dim() == 4 and u.shape[1] == self.num_kv_heads:
                u = rearrange(u, 'b h t d -> b t h d')

            k_eff, u_eff = k, u
            if use_cache and past_key_values is not None:
                k_flat = k.flatten(-2, -1)
                u_flat = u.flatten(-2, -1)
                k_cached_flat, u_cached_flat = past_key_values.update(
                    attn_state=(k_flat, u_flat),
                    layer_idx=self.layer_idx,
                    offset=q_len,
                )['attn_state']
                if cache_has_content:
                    k_eff = rearrange(k_cached_flat, 'b t (h d) -> b t h d', h=self.num_kv_heads)
                    u_eff = rearrange(u_cached_flat, 'b t (h d) -> b t h d', h=self.num_kv_heads)
        else:
            state = past_key_values[self.layer_idx]
            k_cached_flat, u_cached_flat = state['attn_state']
            T_prev = k_cached_flat.shape[1]
            k_prev = rearrange(k_cached_flat, 'b t (h d) -> b t h d', h=self.num_kv_heads)
            u_prev = rearrange(u_cached_flat, 'b t (h d) -> b t h d', h=self.num_kv_heads)

            if attention_mask is not None:
                attn_mask_prev = attention_mask[:, :T_prev]
                q_padded, (k_padded_prev, u_padded_prev), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                    k,
                    (k_prev, u_prev),
                    attn_mask_prev,
                    q_len,
                )
                cu_seqlens_q, cu_seqlens_k = cu_seqlens
                max_seqlen_q, max_seqlen_k = max_seq_lens
                s = flash_attn_varlen_func(
                    q_padded, k_padded_prev, u_padded_prev,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    causal=False,
                    window_size=(-1, -1)
                )
                s = pad_input(s, indices_q, batch_size, q_len)
            else:
                s = flash_attn_func(k, k_prev, u_prev, causal=False, window_size=(-1, -1))

            u_cur = v - rearrange(beta, 'b h t -> b t h 1') * s
            k_eff = torch.cat([k_prev, k], dim=1)
            u_eff = torch.cat([u_prev, u_cur], dim=1)

            past_key_values.update(
                attn_state=(k.flatten(-2, -1), u_cur.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if attention_mask is not None:
            q_padded, (k_padded, u_padded), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                q,
                (k_eff, u_eff),
                attention_mask,
                q_len,
            )
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q_padded, k_padded, u_padded,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1)
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        else:
            if cu_seqlens_rope is not None:
                o = flash_attn_varlen_func(
                    q.squeeze(0), k_eff.squeeze(0), u_eff.squeeze(0),
                    cu_seqlens_q=cu_seqlens_rope,
                    cu_seqlens_k=cu_seqlens_rope,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                    window_size=(-1, -1)
                ).unsqueeze(0)
            else:
                o = flash_attn_func(q, k_eff, u_eff, causal=True, window_size=(-1, -1))

        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        return o, attentions, past_key_values
