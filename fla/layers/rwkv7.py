# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.rwkv6 import LoRA
from fla.modules import GroupNorm
from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7

if TYPE_CHECKING:
    from fla.models.utils import Cache


class RWKV7Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = None,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 16,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        **kwargs
    ) -> RWKV7Attention:
        super().__init__()

        self.mode = mode
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = hidden_size
        if head_dim is not None and num_heads is not None:
            raise ValueError("Cannot specify both `head_dim` and `num_heads`.")
        elif head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        elif num_heads is not None:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = num_heads
        else:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")

        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.layer_idx = layer_idx

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_r = nn.Parameter(torch.empty(1, 1, self.key_dim))
        self.x_w = nn.Parameter(torch.empty(1, 1, self.key_dim))
        self.x_k = nn.Parameter(torch.empty(1, 1, self.key_dim))
        self.x_v = nn.Parameter(torch.empty(1, 1, self.value_dim))
        self.x_a = nn.Parameter(torch.empty(1, 1, self.key_dim))
        self.x_g = nn.Parameter(torch.empty(1, 1, self.value_dim))

        self.k_k = nn.Parameter(torch.empty(1, 1, self.key_dim))
        self.k_a = nn.Parameter(torch.empty(1, 1, self.key_dim))
        self.r_k = nn.Parameter(torch.empty(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=decay_low_rank_dim, activation='tanh')
        self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        self.g_norm = GroupNorm(
            num_groups=self.num_heads,
            num_channels=self.value_dim,
            elementwise_affine=elementwise_affine,
            bias=True,
            eps=self.head_dim*norm_eps
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module, gain=2 ** -2.5)
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

        batch_size, seq_len, hidden_size = hidden_states.shape
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if attention_mask is not None:
            hidden_states = hidden_states.mul(attention_mask[:, -hidden_states.shape[-2]:, None])
        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state['conv_state'].unsqueeze(1)
        else:
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state['conv_state'][0]

        # [batch_size, seq_len, hidden_size]
        delta = shifted - hidden_states
        r = hidden_states + delta * self.x_r
        w = hidden_states + delta * self.x_w
        k = hidden_states + delta * self.x_k
        xv = hidden_states + delta * self.x_v
        a = hidden_states + delta * self.x_a
        g = hidden_states + delta * self.x_g

        r = self.r_proj(r)
        w = -F.softplus(-self.w_lora(w)) - 0.5
        k = self.k_proj(k)
        v = self.v_proj(xv)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        if 'v_state' not in kwargs:
            kwargs['v_state'] = v
        else:
            v = torch.lerp(v, kwargs['v_state'], self.v_lora(xv).sigmoid())
        a = self.a_lora(a).sigmoid()
        g = self.g_lora(r)

        kk = k * self.k_k
        kk = F.normalize(kk.view(batch_size, seq_len, self.num_heads, -1), dim=-1, p=2.0).view(batch_size, seq_len, -1)
        k = k * (1 + (a - 1) * self.k_a)

        r, w, k, v, kk, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', h=self.num_heads), (r, w, k, v, kk, a))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        rwkv7_fn = chunk_rwkv7 if mode == 'chunk' else fused_recurrent_rwkv7
        cu_seqlens = kwargs.get('cu_seqlens', None)
        o, recurrent_state = rwkv7_fn(
            r=r,
            log_w=w,
            k=k,
            v=v,
            a=-kk,
            b=kk * a,
            scale=1.,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
            head_first=False
        )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=hidden_states[:, -1],
                layer_idx=self.layer_idx,
                offset=r.shape[1]
            )

        o = self.g_norm(rearrange(o, '... h d -> ... (h d)'))
        o = o + ((r * k * self.r_k).sum(-1, keepdim=True) * v).view(batch_size, seq_len, -1)
        o = self.o_proj(o * g)

        return o, None, past_key_values
