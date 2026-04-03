# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

from fla.ops.mala import naive_mala_attn
from fla.modules.rotary import RoPE

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

"""
MALA (Magnitude-Aware Linear Attention) implementation.

Based on the paper:
"Rectifying Magnitude Neglect in Linear Attention"
ICCV 2025 (highlight)
https://arxiv.org/abs/2507.00698

Original implementation:
https://github.com/aldjalkdf/MAViT
"""


class MalaAttention(nn.Module):
    r"""
    The layer implementation for MALA (Magnitude-Aware Linear Attention).
    
    Based on the paper:
    "Rectifying Magnitude Neglect in Linear Attention"
    ICCV 2025 (highlight)
    https://arxiv.org/abs/2507.00698
    
    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 1.0.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_kv_heads (int, Optional):
            The number of key/value heads, used for MQA. Default: None.
        use_lepe (bool, Optional):
            Whether to use local positional embedding. Default: `True`.
        lepe_kernel_size (int, Optional):
            The kernel size for local positional embedding. Default: 5.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        use_lepe: bool = True,
        lepe_kernel_size: int = 5,
        layer_idx: int | None = None,
    ) -> MalaAttention:
        super().__init__()

        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_lepe = use_lepe
        self.lepe_kernel_size = lepe_kernel_size
        self.layer_idx = layer_idx

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Projection layers
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        
        # Local positional embedding
        if use_lepe:
            self.lepe = nn.Conv1d(
                in_channels=self.value_dim,
                out_channels=self.value_dim,
                kernel_size=lepe_kernel_size,
                padding=lepe_kernel_size // 2,
                groups=self.value_dim
            )

        # Output gate projection
        self.o_gate_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # RoPE embedding
        self.rope = RoPE(self.head_k_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        batch_size, seq_len, _ = hidden_states.shape

        # Projection
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        o_gate = self.o_gate_proj(hidden_states)

        # Reshape tensors
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_k_dim)
        if self.num_kv_groups > 1:
            k = rearrange(k, 'b s (h d) -> b s h d', d=self.head_k_dim)
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = rearrange(v, 'b s (h d) -> b s h d', d=self.head_v_dim)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)
        else:
            k = rearrange(k, 'b s (h d) -> b s h d', d=self.head_k_dim)
            v = rearrange(v, 'b s (h d) -> b s h d', d=self.head_v_dim)

        # Compute RoPE embeddings
        sin, cos = self.rope(seq_len)
        sin = sin.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, self.num_heads, self.head_k_dim)
        cos = cos.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, self.num_heads, self.head_k_dim)

        # Compute attention
        o = naive_mala_attn(
            q=q,
            k=k,
            v=v,
            sin=sin,
            cos=cos,
        )

        # Apply local positional embedding if enabled
        if self.use_lepe:
            lepe_input = rearrange(v, 'b s h d -> b (h d) s')
            lepe_output = self.lepe(lepe_input)
            lepe_output = rearrange(lepe_output, 'b (h d) s -> b s h d', h=self.num_heads)
            o = o + lepe_output

        # Apply output gate
        o = rearrange(o, 'b s h d -> b s (h d)')
        o = o * o_gate

        # Final projection
        o = self.o_proj(o)

        return o, None, past_key_values
