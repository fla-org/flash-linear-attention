# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Linear attention in Based.
https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py
"""

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules.feature_map import TaylorFeatureMap
from fla.ops.based import parallel_based
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn


class BasedLinearAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        feature_dim: int = 16,
        num_key_value_heads: int = 12,
        num_heads: int = 12,
        feature_name: str = "taylor_exp",
        eps: float = 1e-12,
        causal: bool = True,
        mode: str = "parallel",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.mode = mode
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_key_value_heads
        assert self.hidden_size % self.head_dim == 0
        self.causal = causal

        self.q_proj = nn.Linear(self.hidden_size, self.feature_dim * self.num_heads, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.feature_dim * self.num_heads, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Identity()
        self.feature_map = TaylorFeatureMap(feature_dim)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        mode = self.mode
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q, k, v = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim), [q, k, v])
        if mode == "fused_chunk":
            q, k = self.feature_map(q), self.feature_map(k)
            o, _ = fused_chunk_linear_attn(q, k, v, normalize=True, scale=1)
        elif mode == 'chunk':
            q, k = self.feature_map(q), self.feature_map(k)
            o, _ = chunk_linear_attn(q, k, v, normalize=True, scale=1)
        elif mode == 'parallel':
            assert q.shape[-1] <= 128
            o = parallel_based(q, k, v, scale=1, use_norm=True)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        o = self.dropout(o)
        return o

    def forward_reference(self, hidden_states: torch.Tensor, **kwargs):
        """
        x (torch.Tensor): tensor of shape (b, d, t)
        y (torch.Tensor): tensor of shape (b, d, t)
        """
        # hidden_states = hidden_states.transpose(1, 2)
        b, t, _ = hidden_states.size()
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q = q.view(b, t, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, t, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, t, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Linear attention
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

        # Compute attention
        if self.causal:
            y = ((q * (k * v).cumsum(2)).sum(-1) / ((q * k.cumsum(2)).sum(-1) + self.eps))
        else:
            y = ((q * (k * v).sum(2, True)).sum(-1) / ((q * k.sum(2, True)).sum(-1) + self.eps))
        y = rearrange(y, 'b h t d -> b t (h d)')
        y = self.o_proj(y.to(hidden_states.dtype))
        y = self.dropout(y)
        return y.to(hidden_states.dtype)
