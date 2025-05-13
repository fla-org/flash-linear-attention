# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

""" Implementing the Deepseek Multi Latent Attention (MLA) module. Reference:

https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L328
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from transformers.utils import logging

from fla.models.utils import Cache
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

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
        num_heads: int = 32,
        rope_theta: float = 10000.,
        max_position_embeddings: Optional[int] = None,
        q_lora_rank: Optional[int] = 1536,  # q lora rank is optional, None indicates no q lora
        qk_rope_head_dim: int = 64,
        kv_lora_rank: int = 512,  # following the original Deepseek paper
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        qk_head_dim: Optional[int] = 192,  # qk_nope_head_dim + qk_rope_head_dim
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

        # module meta info
        self.layer_idx = layer_idx

        # attention params info
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_head_dim

        # setup params
        # Deepseek MLA does not support bias
        linear = partial(nn.Linear, bias=False)

        if q_lora_rank is not None:
            self.q_a_proj = linear(hidden_size, q_lora_rank)
            self.q_a_layernorm = RMSNorm(q_lora_rank)
            self.q_b_proj = linear(q_lora_rank, self.num_heads * self.qk_head_dim)
        else:
            self.q_proj = linear(hidden_size, self.num_heads * self.qk_head_dim)

        self.kv_a_proj_with_mqa = linear(
            hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )

        self.o_proj = linear(
            self.num_heads * self.v_head_dim,
            hidden_size,
        )

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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # if attention_mask is not None, this is doing inference
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # prepare q k v projections
        batch_size, q_len = hidden_states.shape[:-1]
        query_shape = (batch_size, q_len, -1, self.qk_head_dim)
        key_shape = (batch_size, q_len, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is not None:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(query_shape).transpose(1, 2)
        else:
            q_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, v = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, q_len, self.qk_rope_head_dim)

        # apply rotary position embedding
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q_pass.shape[2] + seqlen_offset

            if attention_mask is not None:
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q_pass.shape[2] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        cu_seqlens = kwargs.get("cu_seqlens", None)
        q_rot, k_rot = self.rotary(
            q_rot, k_rot, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )

        # TODO: original Deepseek directly caches the final key_states and value states,
        # which actually does not save any memory?
        # if past_key_value is not None:
        #    key_states, v = past_key_value.update(key_states, v, self.layer_idx, cache_kwargs)

        # get and update from cache, then recover k_pass
        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            compressed_kv_cached, k_rot_cached = past_key_values.update(
                attn_state=(compressed_kv, k_rot),
                layer_idx=self.layer_idx,
                offset=q_len,
            )['attn_state']
            if cache_has_content:
                compressed_kv, k_rot = compressed_kv_cached, k_rot_cached
                k_pass, _ = torch.split(compressed_kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)

        # perform attention
        q = torch.cat((q_pass, q_rot), dim=-1)
        k = torch.cat((k_pass, k_rot), dim=-1)

        if self.qk_head_dim != self.v_head_dim:
            v = F.pad(v, [0, self.qk_head_dim - self.v_head_dim])

        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            softmax_scale=self.scaling,
        )

        if self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_values
