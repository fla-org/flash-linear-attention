# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import GroupNorm
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.forgetting_attn.parallel import parallel_forgetting_attn

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class ForgettingAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: Optional[int] = None,
        use_output_gate: bool = False,
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
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.use_output_gate = use_output_gate
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.f_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        if use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = GroupNorm(
                num_groups=self.num_heads,
                hidden_size=self.hidden_size,
                is_rms_norm=True,
            )
            self.k_norm = GroupNorm(
                num_groups=self.num_kv_heads,
                hidden_size=self.kv_dim,
                is_rms_norm=True,
            )

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

        batch_size, q_len, _ = hidden_states.size()

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        f = F.logsigmoid(self.f_proj(hidden_states).float())
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if past_key_values is not None:
            assert cu_seqlens is None, "cu_seqlens should not be provided when past_key_values is not None"
            state = past_key_values.update(
                attn_state=(k, v, f),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )
            k, v, f = state['attn_state']

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if attention_mask is not None:
            q, (k, v, f), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v, f), attention_mask, q_len, keepdim=True)
            _, cu_seqlens_k = cu_seqlens
            cu_seqlens = cu_seqlens_k
            max_seqlen_q, max_seqlen_k = max_seq_lens
            if max_seqlen_q != max_seqlen_k:
                assert max_seqlen_q == 1, "only support q_len == 1 for decoding"
                o = attn_decoding_one_step(q, k, v, f, cu_seqlens=cu_seqlens)
            else:
                o = parallel_forgetting_attn(q, k, v, f, cu_seqlens=cu_seqlens)
        else:
            o = parallel_forgetting_attn(q, k, v, f, cu_seqlens=cu_seqlens)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices_q, batch_size, q_len)
        o = rearrange(o, '... h d -> ... (h d)')
        if self.use_output_gate:
            o = self.g_proj(hidden_states).sigmoid() * o
        o = self.o_proj(o)
        return o, None, past_key_values
