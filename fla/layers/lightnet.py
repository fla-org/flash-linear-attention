# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ["You Only Scan Once: Efficient Multi-dimension Sequential Modeling with LightNet"](https://arxiv.org/abs/2405.21022)

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.modules.fused_norm_gate import rms_norm_swish_gate_linear
from fla.ops.gla import chunk_gla, fused_recurrent_gla

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class LightNetAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        num_heads: Optional[int] = None,
        expand_ratio: Optional[int] = 128,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        gate_low_rank_dim: int = 128,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None
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

        assert mode in ['chunk', 'fused_chunk'], f"Not supported mode `{mode}`."
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
            nn.Linear(gate_low_rank_dim, hidden_size, bias=False)
        )
        self.g_norm = FusedRMSNormGated(
            hidden_size=hidden_size,
            elementwise_affine=elementwise_affine,
            eps=norm_eps
        )
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        q = F.silu(q)
        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_f_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_i_dim)
        # TODO: this 2 steps took huge amount of time, which should be optimized
        z = k.float().logcumsumexp(1)

        if cu_seqlens is not None:
            raise NotImplementedError("LightNet does not support variable-length sequences for now.")
        k, g = torch.exp(k - z).to(k.dtype), (torch.cat((z[:, :1], z[:, :-1]), 1) - z).to(k.dtype)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=g,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=g,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        o = rms_norm_swish_gate_linear(
            rearrange(o, 'b t h d -> b t (h d)'),
            self.g_proj(hidden_states),
            self.g_norm.weight,
            self.g_norm.bias,
            self.o_proj.weight,
            self.o_proj.bias
        )
        return o, None, past_key_values

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_i_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size
