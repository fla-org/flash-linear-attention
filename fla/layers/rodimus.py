# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from transformers.utils import logging

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding, ShortConvolution
from fla.modules.layernorm_gated import RMSNormGated
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


def align_multiple(value, multiple_size=8):
    if value % multiple_size != 0:
        value += multiple_size - (value % multiple_size)
    return value


def autocast_to_fp16(x):
    if x.dtype not in {torch.float16, torch.bfloat16}:
        return x.to(dtype=torch.bfloat16)
    else:
        return x


class RodimusAttention(nn.Module):
    def __init__(
        self,
        block_type: str = 'rodimus',
        mode: str = 'chunk',
        hidden_size: int = 1024,
        input_gate_low_rank: Optional[Union[float, str]] = 'auto',
        expand_ratio: int = 64,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = True,
        norm_eps: float = 1e-5,
        k_norm_eps: Optional[float] = None,
        residual_in_fp32: bool = True,
        layer_idx: int = None
    ):
        super().__init__()

        self.block_type = block_type
        self.mode = mode
        self.hidden_size = hidden_size
        self.d_inner = align_multiple(int(self.hidden_size * 2), 8)

        self.expand_ratio = expand_ratio
        self.input_gate_low_rank = max(self.hidden_size // 64, 16) if input_gate_low_rank == "auto" else input_gate_low_rank

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.norm_eps = norm_eps
        self.k_norm_eps = k_norm_eps if k_norm_eps is not None else 1e-12
        self.mem_size = expand_ratio

        self.residual_in_fp32 = residual_in_fp32
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not supported mode `{mode}`."

        self.gate_proj = nn.Linear(self.hidden_size, self.d_inner, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.d_inner, bias=False)
        self.activation_norm = RMSNormGated(hidden_size=self.d_inner, eps=norm_eps, norm_before_gate=False)
        self.down_proj = nn.Linear(self.d_inner, self.hidden_size, bias=False)

        if use_short_conv:
            self.short_conv = ShortConvolution(
                hidden_size=self.d_inner,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )

        self.residual_weight = nn.Parameter(torch.ones(
            (self.d_inner, ), dtype=torch.float32 if self.residual_in_fp32 else None), requires_grad=True)

        self.k_proj = nn.Linear(self.d_inner, self.mem_size, bias=False)
        self.q_proj = nn.Linear(self.d_inner, self.mem_size, bias=False)

        self.g_gate_proj = nn.Linear(self.d_inner, self.mem_size, bias=True)
        self.tau_gate_proj = nn.Linear(self.d_inner, self.mem_size, bias=True)
        self.i_gate_proj = nn.Sequential(
            nn.Linear(self.d_inner, self.input_gate_low_rank, bias=False),
            nn.Linear(self.input_gate_low_rank, self.d_inner, bias=True),
            nn.Sigmoid(),
        )

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

        batch_size, q_len, _ = hidden_states.shape
        # mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        hidden_states, final_gate = self.up_proj(hidden_states), self.gate_proj(hidden_states)

        if self.use_short_conv:
            conv_state = None
            if last_state is not None:
                conv_state = last_state['conv_state']
            shift_hidden_states, conv_state = self.short_conv(
                x=hidden_states,
                cache=conv_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            shift_hidden_states = hidden_states

        q = self.q_proj(shift_hidden_states)
        k = self.k_proj(shift_hidden_states)
        v = self.i_gate_proj(hidden_states) * hidden_states

        g_gate = F.linear(shift_hidden_states, self.g_gate_proj.weight) + self.g_gate_proj.bias.float()
        tau_gate = F.linear(shift_hidden_states, self.tau_gate_proj.weight) + self.tau_gate_proj.bias.float()

        g_gate = F.softplus(g_gate)
        it_gate = g_gate
        rt_gate_log = -g_gate

        tau_gate = F.sigmoid(tau_gate)
        it_gate = it_gate ** tau_gate
        rt_gate_log = rt_gate_log * tau_gate

        k = F.normalize(k.float(), dim=-1, eps=self.k_norm_eps) * it_gate
        q, k, v, rt_gate_log = map(lambda x: x.unsqueeze(1).transpose(1, 2), (q, k, v, rt_gate_log))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=rt_gate_log,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                head_first=False,
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(
                q=q,
                k=k,
                v=v,
                g=rt_gate_log,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                head_first=False,
            )
        elif mode == 'chunk':
            q, k, rt_gate_log = map(lambda x: x.to(v.dtype), (q, k, rt_gate_log))
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=rt_gate_log,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                head_first=False,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        rodimus_caches = None
        if past_key_values is not None:
            if self.block_type == 'rodimus':
                past_key_values.update(
                    recurrent_state=recurrent_state,
                    conv_state=conv_state if self.use_short_conv else None,
                    layer_idx=self.layer_idx,
                    offset=q_len
                )
            else:
                rodimus_caches = (recurrent_state, conv_state if self.use_short_conv else None)

        o = (o.transpose(1, 2).squeeze(1) + (shift_hidden_states.float()
             if self.residual_in_fp32 else shift_hidden_states) * self.residual_weight).to(o.dtype)

        o = self.activation_norm(o, final_gate)
        o = self.down_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        if self.block_type == 'rodimus':
            return o, None, past_key_values
        else:
            return o, None, (past_key_values, rodimus_caches)


class SlidingWindowSharedKeyAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int = 2048,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        rodimus_caches = kwargs.get('rodimus_caches', None)

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens', None)

        seqlen_offset, max_seqlen = 0, q.shape[1]
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            if rodimus_caches is not None:
                recurrent_state, conv_state = rodimus_caches
            else:
                recurrent_state, conv_state = None, None

            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                attn_state=[k.flatten(-2, -1), v.flatten(-2, -1)],
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        q, k, v = map(autocast_to_fp16, (q, k, v))
        k = repeat(k, "... h d -> ... (n h) d", n=self.num_heads)
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                q=q,
                states=(k, v),
                attention_mask=attention_mask[:, -max(self.window_size, q_len):],
                q_len=q_len
            )
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o.to(dtype=self.o_proj.weight.dtype))

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values
