# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# "Hierarchically Gated Recurrent Neural Network for Sequence Modeling" [https://arxiv.org/abs/2311.04823]

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.modules.activations import swiglu
from fla.ops.hgrn import chunk_hgrn, fused_recurrent_hgrn

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class HGRNAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_ratio: Optional[int] = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None
    ) -> HGRNAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_ratio = expand_ratio
        self.input_dim = int(hidden_size * expand_ratio)

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.i_proj = nn.Linear(hidden_size, self.input_dim, bias=False)
        self.f_proj = nn.Linear(hidden_size, self.input_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.input_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.f_conv1d = ShortConvolution(
                hidden_size=self.input_dim,
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

        self.g_norm = FusedRMSNormGated(
            hidden_size=self.input_dim,
            elementwise_affine=elementwise_affine,
            eps=norm_eps
        )
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

        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if not self.training and hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if self.use_short_conv:
            conv_state_i, conv_state_f = None, None
            if last_state is not None:
                conv_state_i, conv_state_f = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            i, conv_state_i = self.i_conv1d(
                x=self.i_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_i,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            f, conv_state_f = self.f_conv1d(
                x=self.f_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_f,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            i = self.i_proj(hidden_states)
            f = self.f_proj(hidden_states)

        f = F.logsigmoid(f)
        # the lower bound for the first layer is zero
        if lower_bound is not None and self.layer_idx > 0:
            f = torch.logaddexp(lower_bound.log(), torch.log1p(-lower_bound) + f).to(f)
        i = swiglu(i, 1 - f.exp())

        # dealing with left-padding
        if attention_mask is not None:
            i = i.mul_(attention_mask[:, -i.shape[-2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            if cu_seqlens is not None:
                raise NotImplementedError("Chunk mode does not support variable-length sequences.")
            o, recurrent_state = chunk_hgrn(
                x=i,
                g=f,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_hgrn(
                x=i,
                g=f,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_i, conv_state_f) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=i.shape[2]
            )

        o = self.g_norm(o, self.g_proj(hidden_states))
        o = self.o_proj(o)

        return o, None, past_key_values

    def state_size(self, **kwargs) -> int:
        state_size = self.hidden_size
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size
