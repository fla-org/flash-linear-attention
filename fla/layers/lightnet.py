# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# ["You Only Scan Once: Efficient Multi-dimension Sequential Modeling with LightNet"](https://arxiv.org/abs/2405.21022)

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.modules.fused_norm_gate import rms_norm_swish_gate_linear
from fla.ops.lightnet import chunk_lightnet, fused_recurrent_lightnet

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class LightNetAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        num_heads: int | None = None,
        expand_ratio: int | None = 128,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        gate_low_rank_dim: int = 128,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
    ) -> LightNetAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size

        if expand_ratio is None and num_heads is not None:
            expand_ratio = hidden_size // num_heads
        elif expand_ratio is not None and num_heads is None:
            num_heads = hidden_size // expand_ratio
        elif expand_ratio is None and num_heads is None:
            raise RuntimeError("One of `expand_ratio` or `num_heads` should be provided.")
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.key_dim = int(self.num_heads * self.expand_ratio)
        self.value_dim = hidden_size
        self.gate_low_rank_dim = gate_low_rank_dim
        self.layer_idx = layer_idx

        # `fused_chunk` was dropped when `fused_chunk_gla` was deprecated in favour of `chunk_gla`
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_f_dim = self.expand_ratio
        self.head_i_dim = self.hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, hidden_size, bias=False),
        )
        self.g_norm = FusedRMSNormGated(
            hidden_size=hidden_size,
            elementwise_affine=elementwise_affine,
            eps=norm_eps,
        )
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else self.mode

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get('cu_seqlens')
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, cu_seqlens, attention_mask, q_len)
        # with one query per row, B retained tokens imply no padding.
        if indices is not None and q_len == 1 and hidden_states.shape[1] == batch_size:
            hidden_states = hidden_states.reshape(batch_size, 1, self.hidden_size)
            indices, cu_seqlens = None, None
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        q = F.silu(q)
        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_f_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_i_dim)
        if mode == 'fused_recurrent':
            attention_fn = fused_recurrent_lightnet
        elif mode == 'chunk':
            attention_fn = chunk_lightnet
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        initial_state = (last_state['recurrent_state'], last_state.get('ffn_state')) if last_state is not None else None
        o, (recurrent_state, last_z) = attention_fn(
            q=q,
            k=k,
            v=v,
            initial_state=initial_state,
            output_final_state=use_cache,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            ffn_state=last_z,
            offset=q_len,
        )

        o = rearrange(o, 'b t h d -> b t (h d)')
        # the fused norm backward requires a nonempty input.
        if o.shape[1] == 0:
            o = self.o_proj(o)
        else:
            o = rms_norm_swish_gate_linear(
                o,
                self.g_proj(hidden_states),
                self.g_norm.weight,
                self.g_norm.bias,
                self.o_proj.weight,
                self.o_proj.bias,
            )
        o = repad_hidden_states(o, indices, batch_size, q_len)
        return o, None, past_key_values

    def state_size(self, **kwargs) -> int:
        # recurrent_state: [num_heads, head_i_dim, head_f_dim]
        # ffn_state per sequence: [1, num_heads, head_f_dim]
        # num_heads * head_f_dim equals key_dim, so ffn_state contributes self.key_dim elements
        state_size = self.key_dim * self.head_i_dim + self.key_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size
