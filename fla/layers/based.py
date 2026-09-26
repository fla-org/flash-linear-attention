# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules.feature_map import TaylorFeatureMap
from fla.ops.based import parallel_based
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn


class BasedLinearAttention(nn.Module):
    r"""
    Based linear attention with a second-order Taylor feature map.

    Adapted from https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py.

    Args:
        hidden_size (int):
            The hidden size of the input.
        feature_dim (int, Optional):
            The query/key dimension per head before feature mapping. Default: 16.
        num_key_value_heads (int, Optional):
            The number of key/value heads. Must equal `num_heads`. Default: 12.
        num_heads (int, Optional):
            The number of attention heads. Default: 12.
        feature_name (str, Optional):
            The feature map to use. Only `taylor_exp` is supported. Default: `taylor_exp`.
        eps (float, Optional):
            The denominator epsilon for `forward_reference`, also used for noncausal attention. Default: 1e-12.
            Causal kernels use their own normalization epsilon.
        causal (bool, Optional):
            Whether to apply causal attention. Noncausal attention uses the PyTorch reference. Default: `True`.
        mode (str, Optional):
            Which causal kernel to use: `parallel`, `chunk`, or `fused_chunk`. Default: `parallel`.
            The parallel kernel supports `feature_dim` up to 128.
    """

    def __init__(
        self,
        hidden_size: int,
        feature_dim: int = 16,
        num_key_value_heads: int = 12,
        num_heads: int = 12,
        feature_name: str = 'taylor_exp',
        eps: float = 1e-12,
        causal: bool = True,
        mode: str = 'parallel',
    ) -> None:
        super().__init__()

        assert mode in ['parallel', 'chunk', 'fused_chunk'], f"Not supported mode `{mode}`."
        assert feature_name == 'taylor_exp', f"Not supported feature map `{feature_name}`."
        assert hidden_size > 0 and feature_dim > 0, "hidden_size and feature_dim must be positive."
        assert num_heads > 0 and num_key_value_heads > 0, "Head counts must be positive."
        assert num_heads == num_key_value_heads, "num_heads must equal num_key_value_heads."
        assert hidden_size >= num_heads, "hidden_size must be at least num_heads."
        if causal and mode == 'parallel':
            assert feature_dim <= 128, "The parallel kernel only supports feature_dim up to 128."

        self.mode = mode
        self.hidden_size = hidden_size
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_k_dim = feature_dim
        self.head_v_dim = hidden_size // num_heads
        self.head_dim = self.head_v_dim
        self.causal = causal
        self.eps = eps

        self.feature_map = TaylorFeatureMap(head_dim=self.head_k_dim)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_k_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_k_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_v_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_v_dim, hidden_size, bias=False)
        self.dropout = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return attention outputs with shape `[batch_size, seq_len, hidden_size]`."""
        if not self.causal:
            return self.forward_reference(hidden_states=hidden_states)

        mode = self.mode
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if mode == 'parallel':
            o = parallel_based(q=q, k=k, v=v, scale=self.head_k_dim ** -0.5, use_norm=True)
        elif mode in ['chunk', 'fused_chunk']:
            q = self.feature_map(q)
            k = self.feature_map(k)
            if mode == 'chunk':
                o, _ = chunk_linear_attn(q=q, k=k, v=v, scale=1, normalize=True)
            else:
                o, _ = fused_chunk_linear_attn(q=q, k=k, v=v, scale=1, normalize=True)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)
        o = self.dropout(o)
        return o

    def forward_reference(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute normalized Taylor attention in PyTorch, honoring `causal` and `eps`."""
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_k_dim)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_k_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        q = self.feature_map(q).unsqueeze(-2)
        k = self.feature_map(k).unsqueeze(-2)
        v = v.unsqueeze(-1)

        if self.causal:
            o = (q * (k * v).cumsum(1)).sum(-1) / ((q * k.cumsum(1)).sum(-1) + self.eps)
        else:
            o = (q * (k * v).sum(1, True)).sum(-1) / ((q * k.sum(1, True)).sum(-1) + self.eps)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o.to(hidden_states.dtype))
        o = self.dropout(o)
        return o.to(hidden_states.dtype)
