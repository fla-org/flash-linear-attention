# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.rwkv6 import LoRA
from fla.modules import GroupNorm
from fla.modules.l2norm import l2_norm
from fla.modules.token_shift import token_shift
from fla.ops.rwkv7 import chunk_rwkv7, fused_mul_recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7
from fla.ops.rwkv7.gate_output_correction import gate_output_correction

if TYPE_CHECKING:
    from fla.models.utils import Cache


class RWKV7Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = None,
        decay_low_rank_dim: Optional[int] = None,
        gate_low_rank_dim: Optional[int] = None,
        a_low_rank_dim: Optional[int] = None,
        v_low_rank_dim: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        fuse_norm: bool = False,
        value_dim: int = None,
        num_hidden_layers: int = None,
        **kwargs
    ) -> RWKV7Attention:
        super().__init__()

        self.mode = mode
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        if head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        elif head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        elif num_heads is not None:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = num_heads
        self.head_v_dim = int(self.value_dim // self.num_heads)

        # Increase lora dimension for headdim>64
        factor = self.head_dim / 64
        if decay_low_rank_dim is None:
            decay_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
            self.decay_low_rank_dim = decay_low_rank_dim
        else:
            self.decay_low_rank_dim = decay_low_rank_dim

        if gate_low_rank_dim is None:
            gate_low_rank_dim = max(32, int(round((5 * (hidden_size**0.5)) / 32) * 32))
            self.gate_low_rank_dim = gate_low_rank_dim
        else:
            self.gate_low_rank_dim = gate_low_rank_dim

        if a_low_rank_dim is None:
            a_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
            self.a_low_rank_dim = a_low_rank_dim
        else:
            self.a_low_rank_dim = a_low_rank_dim

        if v_low_rank_dim is None:
            v_low_rank_dim = max(32, int(round((1.7 * (hidden_size**0.5)) * factor / 32) * 32))
            self.v_low_rank_dim = v_low_rank_dim
        else:
            self.v_low_rank_dim = v_low_rank_dim

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.fuse_norm = fuse_norm

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=decay_low_rank_dim, activation='tanh')
        if self.layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        if self.fuse_norm:
            self.g_norm = GroupNorm(
                num_groups=self.num_heads,
                hidden_size=self.value_dim,
                elementwise_affine=elementwise_affine,
                eps=self.head_dim*norm_eps,
                bias=True,
            )
        else:
            self.g_norm = nn.GroupNorm(
                num_groups=self.num_heads,
                num_channels=self.value_dim,
                eps=self.head_dim*norm_eps,
                affine=elementwise_affine
            )

        try:
            from transformers.modeling_utils import _init_weights
        except ImportError:
            _init_weights = True
        if _init_weights:
            self.apply(self._initialize_weights)
        for name, module in self.named_modules():
            module._in_rwkv_module = True

        warnings.warn(
            "According to Bo, you are using a potentially buggy FLA implementation of RWKV. "
            "If you plan to report any numbers based on this implementation, we strongly recommend "
            "cross-checking with the official repo: https://github.com/BlinkDL/RWKV-LM. "
            "Bo may disagree with results reported from this version."
        )

    @torch.no_grad()
    @torch.compiler.disable
    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return

        # Initialize only when we're processing the RWKV7Attention module itself
        if isinstance(module, RWKV7Attention) and self.layer_idx is not None:
            ratio_0_to_1 = self.layer_idx / (self.num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)  # 1 to ~0

            # Create position-based initialization tensor
            ddd = torch.ones(1, 1, self.hidden_size, device=self.x_r.device)
            www = torch.zeros(self.hidden_size, device=self.x_r.device)
            zigzag = torch.zeros(self.hidden_size, device=self.x_r.device)
            linear = torch.zeros(self.hidden_size, device=self.x_r.device)
            for n in range(self.hidden_size):
                linear[n] = n / (self.hidden_size-1) - 0.5
                zigzag[n] = ((n % self.head_dim) - ((self.head_dim-1) / 2)) / ((self.head_dim-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (self.hidden_size - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
                ddd[0, 0, n] = n / self.hidden_size

            # Initialize x_* parameters directly
            self.x_r.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_r.dtype)
            self.x_w.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_w.dtype)
            self.x_k.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_k.dtype)
            self.x_v.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_v.dtype)
            self.x_a.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_a.dtype)
            self.x_g.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_g.dtype)

            # Initialize k_k, k_a, r_k
            nn.init.constant_(self.k_a, 1.02)
            nn.init.constant_(self.r_k, -0.04)
            self.k_k.data.copy_((torch.zeros(self.hidden_size, device=self.k_k.device) +
                                0.71 - linear*0.1).to(self.k_k.dtype))
            # Set specific bias values for LoRA modules
            # 0.5 comes from F.softplus
            self.w_lora.set_bias_value(www + 0.5 + zigzag*2.5)
            self.a_lora.set_bias_value(-0.19 + zigzag*0.3 + linear*0.4)

            # v0 initialization - ones (for non-first layers)
            if self.layer_idx != 0:
                self.v_lora._initialize_weights(self.v_lora)
                self.v_lora.set_bias_value(0.73 - linear*0.4)

            # Initialize GroupNorm
            self.g_norm.weight.data[:] = ((self.layer_idx + 1) / self.num_hidden_layers) ** 0.7

            # Initialize Linear projections
            nn.init.orthogonal_(self.r_proj.weight)
            nn.init.orthogonal_(self.k_proj.weight, gain=0.1)
            nn.init.orthogonal_(self.v_proj.weight)
            self.o_proj.weight.data.zero_()

            # Clean up temporary tensors to free memory
            del ddd, www, zigzag, linear

        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        v_first: torch.Tensor = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, seq_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if attention_mask is not None:
            hidden_states = hidden_states.mul(attention_mask[:, -seq_len:, None])

        # delta [batch_size, seq_len, hidden_size]
        if last_state is None:
            delta = token_shift(hidden_states, cu_seqlens)
            recurrent_state = None
        elif hidden_states.shape[1] == 1:
            shifted = last_state['conv_state'].unsqueeze(1)
            delta = shifted - hidden_states
            recurrent_state = last_state['recurrent_state']
        else:
            shifted = self.time_shift(hidden_states)
            shifted[:, 0] = last_state['conv_state']
            delta = shifted - hidden_states
            recurrent_state = last_state['recurrent_state']

        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(hidden_states, delta, self.x_r, self.x_w,
                                                     self.x_k, self.x_v, self.x_a, self.x_g)

        r = self.r_proj(xr)
        # Using bf16 for LoRA computation is numerically safe here because:
        # 1. After sigmoid activation:
        #    - Max absolute error (vs float32): 0.003
        #    - Mean absolute error: 0.0004
        # 2. Subsequent scaling by -0.6065 will further reduce relative error
        #    (error scales linearly with constant multiplication)
        # 3. Final compounded error remains within acceptable bounds for bf16 precision
        # Empirical observation confirms bf16 introduces no practical degradation
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()

        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        if self.fuse_norm:
            kk = l2_norm(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim))
        else:
            kk = F.normalize(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim), dim=-1, p=2.0)

        # Prefer addcmul over expanded form for numerical stability in bf16:
        # 1. Fused Multiply-Add (FMA) in addcmul reduces intermediate rounding:
        #    - Single op vs original 3 ops (mul, sub, mul)
        #    - 1 less intermediate value storage (bf16 write->read overhead)
        # 2. Mathematically equivalent to k*(1 + (a-1)*self.k_a)
        #    but with better precision preservation
        # 3. Particularly crucial for bf16 where intermediate values easily lose precision
        # 4. Pytorch method: k = k.addcmul(k * (a - 1), self.k_a)
        k = fused_k_rwkv7(k, a, self.k_a)

        # dealing with left-padding
        if attention_mask is not None:
            v = v * attention_mask[:, -seq_len:, None]

        r, w, k, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim), (r, w, k, a))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        if self.training or seq_len >= 64:
            # if training, use chunk mode no matter how short the sequence is
            # launching the triton kernel for just one token will actually be slower
            o, recurrent_state = chunk_rwkv7(
                r=r,
                w=w,
                k=k,
                v=v,
                a=-kk,
                b=kk * a,
                scale=1.,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_mul_recurrent_rwkv7(
                r=r,
                w=w,
                k=k,
                v=v,
                kk=kk,
                a=a,
                scale=1.,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=hidden_states[:, -1],
                layer_idx=self.layer_idx,
                offset=r.shape[1]
            )

        if self.fuse_norm:
            o = self.g_norm(rearrange(o, '... h d -> ... (h d)'))
        else:
            o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(batch_size, seq_len, -1)

        o = gate_output_correction(o, r, k, self.r_k, v, g)
        o = self.o_proj(o)

        return o, None, past_key_values, v_first
