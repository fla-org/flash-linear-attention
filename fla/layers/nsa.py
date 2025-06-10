# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.modules import RotaryEmbedding
from fla.ops.nsa.parallel import parallel_nsa
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class NativeSparseAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 64,
        num_kv_heads: Optional[int] = 4,
        head_dim: int = 64,
        qkv_bias: bool = False,
        block_size: Optional[int] = 64,
        block_counts: Optional[Union[torch.LongTensor, int]] = 16,
        window_size: Optional[int] = 512,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias

        self.block_size = block_size
        self.block_counts = block_counts
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads * 3, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

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
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, seq_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=3)
        g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)

        cu_seqlens = kwargs.get('cu_seqlens', None)

        seqlen_offset, max_seqlen = 0, seq_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=seq_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        o = parallel_nsa(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            block_size=self.block_size,
            block_counts=self.block_counts,
            window_size=self.window_size,
            cu_seqlens=cu_seqlens,
        )
        o = o.reshape(batch_size, seq_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values
