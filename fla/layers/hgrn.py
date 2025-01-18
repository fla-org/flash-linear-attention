# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

# "Hierarchically Gated Recurrent Neural Network for Sequence Modeling" [https://arxiv.org/abs/2311.04823]

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedRMSNormSwishGate, ShortConvolution
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

        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."

        self.i_proj = nn.Linear(hidden_size, self.input_dim, bias=False)
        self.f_proj = nn.Linear(hidden_size, self.input_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.input_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.input_dim, conv_size, activation=None)
            self.f_conv1d = ShortConvolution(self.input_dim, conv_size, activation=None)
            self.i_conv1d = ShortConvolution(self.input_dim, conv_size, activation=None)

        self.g_norm = FusedRMSNormSwishGate(self.input_dim, elementwise_affine, norm_eps)
        self.o_proj = nn.Linear(self.input_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

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
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if self.use_short_conv:
            conv_state_i, conv_state_f = None, None
            if last_state is not None:
                conv_state_i, conv_state_f = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)
            i, conv_state_i = self.i_conv1d(x=self.i_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_i,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
            f, conv_state_f = self.f_conv1d(x=self.f_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_f,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
        else:
            i = self.i_proj(hidden_states)
            f = self.f_proj(hidden_states)

        # the lower bound for the first layer is zero
        if lower_bound is None or self.layer_idx == 0:
            i, f = swiglu(i, 1 - f.sigmoid()), F.logsigmoid(f)
        else:
            g = lower_bound + (1 - lower_bound) * f.sigmoid()
            i, f = swiglu(i, 1 - g), g.log()

        # dealing with left-padding
        if attention_mask is not None:
            i = i.mul_(attention_mask[:, -i.shape[-2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_hgrn(i, f, recurrent_state, use_cache)
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_hgrn(i, f, recurrent_state, use_cache, offsets=kwargs.get('offsets', None))
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
