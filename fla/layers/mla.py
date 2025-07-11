# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

""" Implementing the Deepseek Multi Latent Attention (MLA) module. Reference:

https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L328
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MultiheadLatentAttention(nn.Module):
    r"""
    Multi-headed attention from [Deepseek V2](https://arxiv.org/abs/2405.04434)
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        q_lora_rank: Optional[int] = 1536,  # q lora rank is optional, None indicates no q lora
        qk_rope_head_dim: int = 64,
        kv_lora_rank: int = 512,  # following the original Deepseek paper
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        qk_head_dim: Optional[int] = 192,  # qk_nope_head_dim + qk_rope_head_dim
        window_size: Optional[int] = None,
        rope_theta: float = 10000.,
        max_position_embeddings: Optional[int] = None,
        rope_scaling: Optional[dict] = None,
        layer_idx: int = None
    ) -> MultiheadLatentAttention:
        super().__init__()

        # sanity check
        if qk_head_dim is not None:
            assert qk_head_dim == qk_nope_head_dim + qk_rope_head_dim, \
                f"qk_head_dim {qk_head_dim} != qk_nope_head_dim {qk_nope_head_dim} + qk_rope_head_dim {qk_rope_head_dim}"
        else:
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # attention params info
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_head_dim

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        if q_lora_rank is not None:
            self.q_proj = nn.Sequential(
                nn.Linear(hidden_size, q_lora_rank, bias=False),
                RMSNorm(q_lora_rank),
                nn.Linear(q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
            )
        else:
            self.q_proj = nn.Linear(hidden_size, self.num_heads * self.qk_head_dim, bias=False)

        self.k_rope = nn.Linear(hidden_size, self.qk_rope_head_dim, bias=False)
        self.kv_proj = nn.Sequential(
            nn.Linear(hidden_size, self.kv_lora_rank, bias=False),
            RMSNorm(self.kv_lora_rank),
            nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        )

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden_size, bias=False)

        self.scaling = self.qk_head_dim ** (-0.5)
        if rope_scaling is not None:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

        self.rotary = RotaryEmbedding(dim=self.qk_rope_head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if attention_mask is not None, this is doing inference
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # prepare q, k, v
        batch_size, q_len, _ = hidden_states.shape

        q_states = self.q_proj(hidden_states)
        q_states = rearrange(q_states, '... (h d) -> ... h d', d=self.qk_head_dim)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pass, k_rot = self.kv_proj(hidden_states), self.k_rope(hidden_states)

        k_rot = rearrange(k_rot, 'b t d -> b t 1 d')
        k_pass = rearrange(k_pass, '... (h d) -> ... h d', d=self.qk_nope_head_dim + self.v_head_dim)
        k_pass, v = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # apply rotary position embedding
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q_len + seqlen_offset

            if attention_mask is not None:
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q_len + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        cu_seqlens = kwargs.get("cu_seqlens", None)
        q_rot, k_rot = self.rotary(
            q_rot, k_rot, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )

        k_rot = repeat(k_rot, 'b t 1 d -> b t h d', h=self.num_heads)
        q = torch.cat((q_pass, q_rot), dim=-1)
        k = torch.cat((k_pass, k_rot), dim=-1)

        # TODO: instead of caching the full k, v, we can actually only cache the compressed_kv and k_rot
        # and recover the full k, v from compressed_kv and k_rot
        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k, v),
                layer_idx=self.layer_idx,
                offset=q_len,
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached

        # Head dim match to use flash-attn
        if self.qk_head_dim != self.v_head_dim:
            v = F.pad(v, [0, self.qk_head_dim - self.v_head_dim])

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )

        if self.qk_head_dim != self.v_head_dim:
            o = o[:, :, :, :self.v_head_dim]
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)
        return o, None, past_key_values
