# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
https://github.com/corl-team/rebased/blob/main/flash_linear_attention/fla/layers/rebased_fast.py
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules.feature_map import RebasedFeatureMap
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn
from fla.ops.rebased import parallel_rebased


class ReBasedLinearAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        l_max: int = 2048,
        feature_dim: int = 16,
        num_key_value_heads: int = 16,
        num_heads: int = 16,
        use_gamma: Optional[bool] = True,
        use_beta: Optional[bool] = True,
        normalize: Optional[bool] = True,
        causal: bool = True,
        eps: float = 1e-5,
        mode: str = "parallel",
        layer_idx: Optional[int] = None,
        **kwargs
    ) -> ReBasedLinearAttention:
        super().__init__()
        self.hidden_size = hidden_size
        self.l_max = l_max
        self.mode = mode
        assert self.mode in ["fused_chunk", "parallel", 'chunk']

        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_key_value_heads
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.normalize = normalize
        self.causal = causal
        self.eps = eps
        self.mode = mode
        self.layer_idx = layer_idx

        self.feature_map = RebasedFeatureMap(self.feature_dim, use_gamma, use_beta, normalize)
        self.q_proj = nn.Linear(self.hidden_size, self.feature_dim * self.num_heads, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.feature_dim * self.num_heads, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        mode = self.mode
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q, k, v = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim), [q, k, v])
        q, k = self.feature_map(q, flatten=(mode != 'parallel')), self.feature_map(k, flatten=(mode != 'parallel'))
        if mode == "fused_chunk":
            o = fused_chunk_linear_attn(
                q=q,
                k=k,
                v=v,
                normalize=True,
                scale=1,
            )
        elif mode == 'chunk':
            o = chunk_linear_attn(
                q=q,
                k=k,
                v=v,
                normalize=True,
                scale=1,
            )
        elif mode == 'parallel':
            assert q.shape[-1] <= 128
            o = parallel_rebased(
                q=q,
                k=k,
                v=v,
                eps=self.eps,
                use_scale=True,
                use_normalize=True,
            )
        o = self.o_proj(o)
        o = self.dropout(o)
        return o

    # https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py#L119
    def forward_reference(
        self,
        hidden_states: torch.Tensor,
        filters: torch.Tensor = None,
        *args,
        **kwargs
    ):
        """
        x (torch.Tensor): tensor of shape (b, d, t)
        y (torch.Tensor): tensor of shape (b, d, t)
        """
        b, t, _ = hidden_states.size()
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q = q.view(b, t, -1, self.feature_dim).transpose(1, 2)
        k = k.view(b, t, -1, self.feature_dim).transpose(1, 2)
        v = v.view(b, t, -1, self.head_dim).transpose(1, 2)

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
