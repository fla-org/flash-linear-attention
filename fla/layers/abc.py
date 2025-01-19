# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules import (FusedRMSNormSwishGate, RMSNorm, RotaryEmbedding,
                         ShortConvolution)
from fla.modules.activations import swiglu, swish
from fla.ops.abc.chunk import chunk_abc

if TYPE_CHECKING:
    from fla.models.utils import Cache


class ABCAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_slots: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        use_input_gate: bool = False,
        use_output_gate: bool = True,
        use_norm: bool = True,
        clamp_min: Optional[float] = -32,
        clamp_max: Optional[float] = 32,
        layer_idx: Optional[int] = None,
        **kwargs
    ) -> ABCAttention:
        super().__init__()

        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.key_dim = int(self.hidden_size * self.expand_k)
        self.value_dim = int(self.hidden_size * self.expand_v)
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.gate_low_rank_dim = gate_low_rank_dim
        self.gate_logit_normalizer = gate_logit_normalizer

        self.use_input_gate = use_input_gate
        self.use_output_gate = use_output_gate
        self.use_norm = use_norm

        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots

        self.norm_eps = norm_eps

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.layer_idx = layer_idx

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        if use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.s_proj = nn.Linear(self.hidden_size, self.num_heads * self.num_slots, bias=False)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        if self.use_norm:
            if self.use_output_gate:
                self.g_norm = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            else:
                self.g_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)

        if self.use_rope:
            self.rotary = RotaryEmbedding(self.head_k_dim)

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
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_q,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_v,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        if self.use_input_gate:
            q, k, v = map(lambda x: swish(x), (q, k, v))
        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        q, k, v = map(lambda x: rearrange(x, '... (h d) -> ... h d', h=self.num_heads), (q, k, v))
        if self.use_rope:
            seqlen_offset = 0
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            q, k = self.rotary(q, k, seqlen_offset=seqlen_offset)

        s = rearrange(self.s_proj(hidden_states), '... (h m) -> ... h m', h=self.num_heads)
        s = s.clamp_(self.clamp_min, self.clamp_max)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        o, recurrent_state = chunk_abc(
            q=q,
            k=k,
            v=v,
            s=s,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            head_first=False
        )
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        if self.use_norm and not self.use_output_gate:
            o = self.g_norm(o)
        elif self.use_output_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
            o = self.g_norm(o, g) if self.use_norm else swiglu(g, o)
        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values

    def state_size(self, seq_len: int = 2048):
        return self.num_heads * self.key_dim * self.head_v_dim
