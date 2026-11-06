# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright (c) 2026, Anamaria-Roberta Hartl, Pieter-Jan Hoedt
# Adapted from the xLSTM repo: https://github.com/NX-AI/xlstm/tree/main/xlstm/blocks/mlstm
# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import GroupLinear, GroupNorm, ShortConvolution
from fla.ops.mlstm import chunk_mlstm, fused_recurrent_mlstm

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


def _round_proj_up_dim(
    hidden_size: int,
    proj_factor: float,
    multiple_of: int = 64,
    round_up: bool = True,
) -> int:
    value = proj_factor * hidden_size / multiple_of
    value = math.ceil(value) if round_up else math.floor(value)
    return int(value * multiple_of)


class MLSTM(nn.Module):
    """
    mLSTM layer implementation.

    Based on [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) by Beck et al. (2024).

    mLSTM uses a matrix-valued recurrent memory with input and forget gates.
    The layer combines a short convolution, grouped q/k/v projections, mLSTM recurrence, and output skip connections.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input and output. Default: 1024.
        num_heads (int, Optional):
            The number of recurrent heads. Must divide the inner hidden size. Default: 16.
        proj_factor (float, Optional):
            The expansion ratio for the inner hidden size, before rounding. Default: 2.0.
        mode (str, Optional):
            Which mLSTM kernel to use: `chunk` or `fused_recurrent`. Training requires `chunk`.
            Single-token inference automatically uses `fused_recurrent`. Default: `chunk`.
        qkv_proj_blocksize (int, Optional):
            The input and output width of each q/k/v projection group.
            Must divide the inner hidden size. Default: 4.
        conv_size (int, Optional):
            The kernel size of the short convolution. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution. Default: `True`.
        bias (bool, Optional):
            Whether to use bias in the up, down, and q/k/v projections.
            Input and forget gate projections always use bias. Default: `False`.
        dropout (float, Optional):
            The dropout probability applied after the down projection. Default: 0.0.
        elementwise_affine (bool, Optional):
            Whether to learn a scale in the per-head output normalization. Default: `True`.
        norm_eps (float, Optional):
            The epsilon value for the output normalization. Default: 1e-5.
        layer_idx (int, Optional):
            The index used to read and update this layer's cache. Default: `None`.
        num_blocks (int, Optional):
            The number of blocks used to scale the down projection during model initialization, at least 1. Default: 1.
        round_proj_up_to_multiple_of (int, Optional):
            The multiple used to round `hidden_size * proj_factor` to the inner hidden size. Default: 64.
        round_proj_up_dim_up (bool, Optional):
            Whether to round the inner hidden size up rather than down. Default: `True`.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        proj_factor: float = 2.0,
        mode: str = 'chunk',
        qkv_proj_blocksize: int = 4,
        conv_size: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
        dropout: float = 0.0,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
        num_blocks: int = 1,
        round_proj_up_to_multiple_of: int = 64,
        round_proj_up_dim_up: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.proj_factor = proj_factor
        self.qkv_proj_blocksize = qkv_proj_blocksize
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.bias = bias
        self.dropout_p = dropout
        self.layer_idx = layer_idx
        self.num_blocks = max(num_blocks, 1)

        self.inner_hidden_size = _round_proj_up_dim(
            hidden_size=hidden_size,
            proj_factor=proj_factor,
            multiple_of=round_proj_up_to_multiple_of,
            round_up=round_proj_up_dim_up,
        )
        if self.inner_hidden_size % num_heads != 0:
            raise ValueError(
                f"`inner_hidden_size` ({self.inner_hidden_size}) must be divisible by `num_heads` ({num_heads}).",
            )
        if self.inner_hidden_size % qkv_proj_blocksize != 0:
            raise ValueError(
                f"`inner_hidden_size` ({self.inner_hidden_size}) must be divisible by "
                f"`qkv_proj_blocksize` ({qkv_proj_blocksize}).",
            )

        self.proj_up = nn.Linear(hidden_size, 2 * self.inner_hidden_size, bias=bias)
        self.num_proj_heads = self.inner_hidden_size // qkv_proj_blocksize
        self.q_proj = GroupLinear(
            in_features=self.inner_hidden_size,
            out_features=self.inner_hidden_size,
            groups=self.num_proj_heads,
            bias=bias,
        )
        self.k_proj = GroupLinear(
            in_features=self.inner_hidden_size,
            out_features=self.inner_hidden_size,
            groups=self.num_proj_heads,
            bias=bias,
        )
        self.v_proj = GroupLinear(
            in_features=self.inner_hidden_size,
            out_features=self.inner_hidden_size,
            groups=self.num_proj_heads,
            bias=bias,
        )
        self.conv1d = ShortConvolution(
            hidden_size=self.inner_hidden_size,
            kernel_size=conv_size,
            bias=conv_bias,
            activation="silu",
        )
        self.igate = nn.Linear(3 * self.inner_hidden_size, num_heads, bias=True)
        self.fgate = nn.Linear(3 * self.inner_hidden_size, num_heads, bias=True)
        self.o_norm = GroupNorm(
            num_groups=num_heads,
            hidden_size=self.inner_hidden_size,
            elementwise_affine=elementwise_affine,
            bias=False,
            eps=norm_eps,
            is_rms_norm=False,
        )
        self.learnable_skip = nn.Parameter(torch.ones(self.inner_hidden_size))
        self.proj_down = nn.Linear(self.inner_hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        del output_attentions

        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "Expected `attention_mask` with shape [batch_size, seq_len] for padding, "
                f"got {tuple(attention_mask.shape)}.",
            )

        batch_size, seq_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if (seq_len == 1 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)
        hidden_states, indices, cu_seqlens = unpad_hidden_states(
            hidden_states=hidden_states,
            cu_seqlens=kwargs.get('cu_seqlens'),
            attention_mask=attention_mask,
            q_len=seq_len,
        )

        conv_state = last_state['conv_state'] if last_state is not None else None
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        x_inner = self.proj_up(hidden_states)
        x, z = torch.split(x_inner, self.inner_hidden_size, dim=-1)

        x_conv, conv_state = self.conv1d(
            x=x,
            cache=conv_state,
            output_final_state=use_cache or past_key_values is not None,
            cu_seqlens=cu_seqlens,
        )

        q = self.q_proj(x_conv)
        k = self.k_proj(x_conv)
        v = self.v_proj(x)

        qkv = torch.cat([q, k, v], dim=-1)
        q, k, v = map(lambda x: rearrange(x, '... (h d) -> ... h d', h=self.num_heads), (q, k, v))
        i = self.igate(qkv)
        f = self.fgate(qkv)

        if mode == 'chunk':
            o, recurrent_state = chunk_mlstm(
                q=q,
                k=k,
                v=v,
                i=i,
                f=f,
                initial_state=recurrent_state,
                output_final_state=use_cache or past_key_values is not None,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_mlstm(
                q=q,
                k=k,
                v=v,
                i=i,
                f=f,
                initial_state=recurrent_state,
                output_final_state=use_cache or past_key_values is not None,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Unsupported mode `{mode}`.")

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_norm(o)

        o = o + self.learnable_skip * x_conv
        o_gated = o * F.silu(z)
        output = self.dropout(self.proj_down(o_gated))

        update_layer_cache(
            module=self,
            past_key_values=past_key_values,
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            offset=seq_len,
        )
        output = repad_hidden_states(output, indices, batch_size, seq_len)

        return output, None, past_key_values

    def state_size(self, **kwargs) -> int:
        del kwargs
        head_dim = self.inner_hidden_size // self.num_heads
        recurrent_size = self.num_heads * (head_dim * head_dim + head_dim + 1)
        return recurrent_size + self.conv1d.state_size
