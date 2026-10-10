# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

from fla.layers.utils import repad_hidden_states, unpad_hidden_states
from fla.modules import RMSNorm
from fla.ops.stickbreaking_attn import parallel_stickbreaking_attn

if TYPE_CHECKING:
    from fla.models.utils import Cache


class StickBreakingAttention(nn.Module):
    """Stick-breaking attention with dense or packed inputs and no positional embeddings.

    Args:
        hidden_size (int, Optional):
            Hidden size. Default: 2048.
        num_heads (int, Optional):
            Number of query heads. Default: 32.
        num_kv_heads (int, Optional):
            Number of key/value heads. Default: `None`, using `num_heads`.
        qkv_bias (bool, Optional):
            Whether the query/key/value projections have biases. Default: `False`.
        qk_norm (bool, Optional):
            Whether to normalize each query/key head. Default: `False`.
        attend_current (bool, Optional):
            Whether to include the key at the query's position. Default: `False`.
        norm_eps (float, Optional):
            Epsilon for query/key normalization. Default: 1e-6.
        layer_idx (int, Optional):
            Layer index. Default: `None`.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attend_current: bool = False,
        norm_eps: float = 1e-6,
        layer_idx: int | None = None,
    ):
        super().__init__()
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by a positive num_heads.")
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by a positive num_kv_heads.")
        if hidden_size // num_heads > 256:
            raise ValueError("The attention head dimension cannot be larger than 256.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.attend_current = attend_current
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if use_cache or past_key_values is not None:
            raise NotImplementedError("Stick-breaking attention does not support KV-cache decoding. Use use_cache=False.")
        if output_attentions:
            raise NotImplementedError("Stick-breaking attention does not return attention matrices.")
        if hidden_states.device.type not in ('cuda', 'xpu'):
            raise NotImplementedError("Stick-breaking attention requires a CUDA/HIP or Intel GPU.")
        batch_size, q_len, _ = hidden_states.shape
        if attention_mask is not None and attention_mask.shape != (batch_size, q_len):
            raise ValueError("attention_mask must have shape [batch_size, sequence_length].")
        cu_seqlens = kwargs.get('cu_seqlens')
        if cu_seqlens is not None and batch_size != 1:
            raise ValueError("Packed inputs with cu_seqlens require batch size 1.")
        hidden_states, indices, cu_seqlens = unpad_hidden_states(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            q_len=q_len,
        )
        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
            raise TypeError(
                "Stick-breaking projections require matching fp16/bf16 tensors. Use mixed precision or convert the model.")
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        o, _ = parallel_stickbreaking_attn(q=q, k=k, v=v, attend_current=self.attend_current, cu_seqlens=cu_seqlens)
        o = self.o_proj(o.flatten(-2))
        o = repad_hidden_states(hidden_states=o, indices=indices, batch_size=batch_size, q_len=q_len)
        return o, None, None
