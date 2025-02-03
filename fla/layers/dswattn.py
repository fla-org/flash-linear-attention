# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from transformers.utils import logging

from fla.modules import RMSNorm, RotaryEmbedding

if TYPE_CHECKING:
    from fla.models.utils import Cache

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

logger = logging.get_logger(__name__)

def generate_alibi_slopes(head_num=4, device=None):
    slopes = torch.pow(2, -8.0 * torch.arange(head_num, device=device) / head_num)
    return slopes

def build_alibi_tensor(head_num: int, seq_len: int, device=None) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device)
    slopes = torch.pow(2, -8.0 * torch.arange(head_num, device=device) / head_num)
    relative_pos = pos[None, :] - pos[:, None]  # shape: (seq_len, seq_len)
    alibi = slopes[:, None, None] * relative_pos[None, :, :]
    return alibi

def build_swa_mask(window_len: int, seq_len: int, device=None) -> torch.Tensor:
    tril = torch.tril(torch.ones(seq_len, seq_len))
    minus = torch.tril(torch.ones(seq_len, seq_len), diagonal=-window_len)
    mask = tril - minus
    return mask.to(device)

class DualSlidingWindowAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        window_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        layer_idx: int = None
    ):
        super().__init__()

        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.norm_first = norm_first
        self.layer_idx = layer_idx

        if norm_first:
            self.norm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.ssm_k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.ssm_v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.alibi_slopes = generate_alibi_slopes(self, num_heads)
        # block_mask for FlexAttention
        def sliding_window_causal(b, h, q_idx, kv_idx):
            is_states = kv_idx < self.seq_len
            causal_mask_s = q_idx >= kv_idx
            window_mask_s = q_idx - kv_idx < window_size
            causal_mask_a = q_idx >= (kv_idx - self.seq_len)
            window_mask_a = q_idx - (kv_idx - self.seq_len) < window_size
            return (is_states & causal_mask_s & window_mask_s) | (~is_states & causal_mask_a & window_mask_a)
        self.swa_mask = create_block_mask(sliding_window_causal, B=None, H=None, Q_LEN=max_position_embeddings, KV_LEN=max_position_embeddings*2)
        self.alibi_tensor = build_alibi_tensor(num_heads, max_position_embeddings) # this one is for inference
        self.mask_tensor = build_swa_mask(window_size, max_position_embeddings)
        self.k_len = max_position_embeddings

    # score_mod for FlexAttention
    def _alibi(self, score, b, h, q_idx, kv_idx):
        bias = self.alibi_bias[h] * ((kv_idx - self.k_len - q_idx) * (kv_idx >= self.k_len) + (kv_idx - q_idx) * (kv_idx < self.k_len))
        return score + bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        ssm_states: torch.Tensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        cache_params = None,
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

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        q = rearrange(self.q_proj(hidden_states), '... t (h d) -> ... h t d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), '... t (h d) -> ... h t d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), '... t (h d) -> ... h t d', h=self.num_kv_heads)
        ssm_k = rearrange(self.ssm_k_proj(ssm_states), '... t (h d) -> ... h t d', h=self.num_kv_heads)
        ssm_v = rearrange(self.ssm_v_proj(ssm_states), '... t (h d) -> ... h t d', h=self.num_kv_heads)

        if cache_params is not None:
            k, v, ssm_k, ssm_v = cache_params.update_att_state(
                layer_idx=self.layer_idx,
                attn_keys=k,
                attn_values=v,
                ssm_keys=ssm_k,
                ssm_values=ssm_v,
            )

        self.k_len = k.size(2)
        # Concatenate SSM keys and values to the usual keys and values
        k, v = torch.cat([ssm_k, k], dim=2), torch.cat([ssm_v, v], dim=2)

        # Use FlexAttention
        if q_len == self.max_position_embeddings: # just in training for now
            o = flex_attention(q, k, v, score_mod=self._alibi, block_mask=self.swa_mask, enable_gqa=True)
        else:
            alibi_tensor = self.alibi_tensor[:, self.k_len-q_len:self.k_len, :self.k_len].to(hidden_states.device)
            mask_tensor = self.mask_tensor[self.k_len-q_len:self.k_len, :self.k_len].to(hidden_states.device)
            # TODO: manual attention

        o = rearrange(o, '... h t d -> ... t (h d)')
        o = self.o_proj(o)

        return o, cache_params
