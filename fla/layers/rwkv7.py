# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from fla.layers.rwkv6 import LoRA
from fla.modules import GroupNorm
from fla.modules.l2norm import l2_norm
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

        # Always register parameters in float32 for precision
        self.register_parameter('x_x', nn.Parameter(torch.zeros(6, hidden_size, dtype=torch.float32)))
        self.register_parameter('k_k', nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32)))
        self.register_parameter('k_a', nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32)))
        self.register_parameter('r_k', nn.Parameter(torch.zeros(self.num_heads, self.head_dim, dtype=torch.float32)))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=decay_low_rank_dim, activation='tanh')
        if self.layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        self.g_norm = GroupNorm(
            num_groups=self.num_heads,
            hidden_size=self.value_dim,
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
        v_first: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # If we're in half-precision mode, temporarily use float32 for all calculations
        original_dtype = hidden_states.dtype
        use_fp32 = original_dtype != torch.float32
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # Cast to float32 for better precision if needed
        if use_fp32:
            hidden_states = hidden_states.float()
            if attention_mask is not None:
                attention_mask = attention_mask.float()
            if v_first is not None:
                v_first = v_first.float()

        batch_size, seq_len, _ = hidden_states.shape

        if self.training:
            # if training, use chunk mode no matter how short the sequence is
            mode = 'chunk'
        else:
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
                shifted[:, 0] = last_state['conv_state']

        # [batch_size, seq_len, hidden_size]
        delta = shifted - hidden_states
        # Cast x_x to float32 for better precision in addcmul
        x_x_float = self.x_x.float()
        xr, xw, xk, xv, xa, xg = hidden_states.addcmul(delta, x_x_float.view(6, 1, 1, -1)).unbind(0)

        r = self.r_proj(xr)
        w = -math.exp(-0.5) * self.w_lora(xw).to(torch.float).sigmoid()
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        elif v_first is not None:
            # Cast to float for higher precision in sigmoid and lerp operations
            v = torch.lerp(v, v_first, self.v_lora(xv).to(torch.float).sigmoid())
        # Cast to float for higher precision in sigmoid operation
        a = self.a_lora(xa).to(torch.float).sigmoid()
        g = self.g_lora(xg)

        # Use float32 for L2 normalization and parameter operations
        k_k_float = self.k_k.float()
        kk = l2_norm((k * k_k_float).view(batch_size, seq_len, self.num_heads, -1)).view(batch_size, seq_len, -1)
        k = k.addcmul(k * (a - 1), self.k_a.float())

        # dealing with left-padding
        if attention_mask is not None:
            v = v * attention_mask[:, -v.shape[-2]:, None]
        r, w, k, v, kk, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim), (r, w, k, v, kk, a))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        rwkv7_fn = chunk_rwkv7 if mode == 'chunk' else fused_recurrent_rwkv7
        cu_seqlens = kwargs.get('cu_seqlens', None)
        
        # Convert tensors used in the core algorithm to float32 for better precision
        r_float = r.float() if r.dtype != torch.float32 else r
        w_float = w.float() if w.dtype != torch.float32 else w
        k_float = k.float() if k.dtype != torch.float32 else k
        v_float = v.float() if v.dtype != torch.float32 else v
        kk_float = kk.float() if kk.dtype != torch.float32 else kk
        a_float = a.float() if a.dtype != torch.float32 else a
        
        o, recurrent_state = rwkv7_fn(
            r=r_float,
            w=w_float,
            k=k_float,
            v=v_float,
            a=-kk_float,
            b=kk_float * a_float,
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
        o_reshaped = rearrange(o, '... h d -> ... (h d)')

        o = self.g_norm(o_reshaped.float())

        # Use float32 for residual calculation for better precision
        r_k_float = self.r_k.float()
        residual = ((r * k * r_k_float).sum(-1, keepdim=True) * v).view(batch_size, seq_len, -1)
        o = o + residual
        o = self.o_proj(o * g)

        # Cast back to original dtype if needed
        if use_fp32:
            o = o.to(original_dtype)
            
        return o, None, past_key_values, v_first