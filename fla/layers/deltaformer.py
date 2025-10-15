# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.deltaformer import deltaformer_attn
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class DeltaFormerAttention(nn.Module):

    r"""
    The layer implementation for DeltaFormer,
    [Understanding Transformer from the Perspective of Associative Memory]
    (https://arxiv.org/pdf/2505.19488).

    Notes
        - DeltaFormer attention is implemented with Triton kernels in `fla.ops.deltaformer` and is tuned
          for typical head dimensions (e.g., 64/128). It currently supports fixed-length inputs.
        - For variable-length inputs (padding masks), the deltaformer computation falls back to using the
          fixed-length path, while the second stage (softmax attention over U) uses FlashAttention's
          varlen path when an attention mask is provided.
        - K/V grouping (GQA) is supported natively by FlashAttention via `num_kv_heads`.
        - Uses K-K similarity in deltaformer computation instead of Q-K similarity for better performance.

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
        rope_theta (float, Optional):
            The base frequency for rotary position embedding. Default: 10000.
        max_position_embeddings (int, Optional):
            The maximum position embeddings. Default: None.
        layer_idx (int, Optional):
            The index of the layer (used for cache compatibility). Default: None.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        rope_theta: float = 10000.,
        max_position_embeddings: int | None = None,
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
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
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
        beta = self.b_proj(hidden_states)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        cu_seqlens_kw = kwargs.get('cu_seqlens')
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q_len + seqlen_offset

            if attention_mask is not None:
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q_len + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens_kw)

        o = deltaformer_attn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens_kw,
        )

        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values
