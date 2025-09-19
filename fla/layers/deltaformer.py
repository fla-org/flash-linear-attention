# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm
from fla.ops.deltaformer import delta_pre_attn

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

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

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

        if self.num_kv_groups > 1:
            k = repeat(k, 'b t h d -> b t (h g) d', g=self.num_kv_groups)
            v = repeat(v, 'b t h d -> b t (h g) d', g=self.num_kv_groups)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if attention_mask is not None:
            # Use varlen FlashAttention path. Pre-attention currently supports fixed length only → fallback by padding.
            q_full = q
            k_full = k
            v_full = v
            beta_full = beta
        else:
            q_full, k_full, v_full, beta_full = q, k, v, beta

        # Compute u via DeltaFormer pre-attention (fixed-length kernel).
        u = delta_pre_attn(
            rearrange(q_full, 'b t h d -> b h t d'),
            rearrange(k_full, 'b t h d -> b h t d'),
            rearrange(v_full, 'b t h d -> b h t d'),
            beta_full,
        )
        u = rearrange(u, 'b h t d -> b t h d')

        # Second stage: standard FlashAttention but using u as values
        if attention_mask is not None:
            q_padded, (k_padded, u_padded), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, u), attention_mask, q_len)
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
            o = flash_attn_func(q, k, u, causal=True, window_size=(-1, -1))

        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        return o, attentions, past_key_values
