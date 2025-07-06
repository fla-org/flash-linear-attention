# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# "HGRN2: Gated Linear RNNs with State Expansion"[https://arxiv.org/abs/2404.07904]

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.activations import swish
from fla.modules.layernorm import rms_norm_linear
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class HGRN2Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        num_heads: Optional[int] = None,
        expand_ratio: Optional[int] = 128,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None
    ) -> HGRN2Attention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size

        if expand_ratio is not None:
            num_heads = hidden_size // expand_ratio
        elif expand_ratio is None and num_heads is not None:
            expand_ratio = hidden_size // num_heads
        elif expand_ratio is None and num_heads is None:
            raise RuntimeError("One of `expand_ratio` or `num_heads` should be provided.")
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.forget_dim = int(self.num_heads * self.expand_ratio)
        self.input_dim = hidden_size
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not supported mode `{mode}`."
        assert self.forget_dim % num_heads == 0, f"forget dim must be divisible by num_heads of {num_heads}"
        assert self.input_dim % num_heads == 0, f"input dim must be divisible by num_heads of {num_heads}"

        self.head_f_dim = self.expand_ratio
        self.head_i_dim = self.hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, self.forget_dim, bias=False)
        self.f_proj = nn.Linear(hidden_size, self.forget_dim, bias=False)
        self.i_proj = nn.Linear(hidden_size, self.input_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.forget_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.f_conv1d = ShortConvolution(
                hidden_size=self.forget_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.i_conv1d = ShortConvolution(
                hidden_size=self.input_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )

        self.g_norm = RMSNorm(hidden_size=self.hidden_size, elementwise_affine=elementwise_affine, eps=norm_eps)
        self.o_proj = nn.Linear(self.input_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_f, conv_state_i = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_f, conv_state_i = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            f, conv_state_f = self.f_conv1d(
                x=self.f_proj(hidden_states),
                cache=conv_state_f,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            i, conv_state_i = self.i_conv1d(
                x=self.i_proj(hidden_states),
                cache=conv_state_i,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = self.q_proj(hidden_states)
            f = self.f_proj(hidden_states)
            i = self.i_proj(hidden_states)

        q = swish(q)

        g = F.logsigmoid(f)
        # the lower bound for the first layer is zero
        if lower_bound is not None and self.layer_idx > 0:
            g = torch.logaddexp(lower_bound.log(), torch.log1p(-lower_bound) + g)
        k = 1 - g.exp()

        q, k, g = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_f_dim), (q, k.to(i), g))
        i = rearrange(i, '... (h d) -> ... h d', d=self.head_i_dim)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=i,
                gk=g,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(
                q=q,
                k=k,
                v=i,
                g=g,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=i,
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
                conv_state=(conv_state_q, conv_state_f, conv_state_i) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        o = rearrange(o, '... h d -> ... (h d)')
        o = rms_norm_linear(o, self.g_norm.weight, self.g_norm.bias, self.o_proj.weight, self.o_proj.bias)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
