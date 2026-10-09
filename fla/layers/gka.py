# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.gka import chunk_gka, fused_recurrent_gka

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class GatedKalmaNet(nn.Module):
    """
    Gated KalmaNet (GKA) layer implementation.

    Reference: `Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression <https://arxiv.org/abs/2511.21016>`_

    At every token t, GKA computes the solution of a ridge regression over the whole history,
    `S_t = argmin_S lamb_t ||S||_F^2 + sum_{i <= t} eta_{t,i} ||S k_i - v_i||^2`,
    with exponentially decaying weights `eta_{t,i}`. The solution is `S_t = U_t^T (H_t + lamb_t I)^{-1}`,
    where `H_t` and `U_t` are the decayed sums of `k_i k_i^T` and `k_i v_i^T`.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value head dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each query/key head. Default: 128.
        num_heads (int, Optional):
            The number of query heads, and of key heads unless `num_kv_heads` is set. Default: 16.
        num_v_heads (int, Optional):
            The number of value heads (GVA), a multiple of `num_heads`. Equal to `num_heads` if `None`.
            Cannot be combined with `num_kv_heads`. Default: `None`.
        num_kv_heads (int, Optional):
            The number of key and value heads (GQA), a divisor of `num_heads`. Equal to `num_heads` if `None`.
            Default: `None`.
        mode (str, Optional):
            Which kernel to use without autograd, `chunk` or `fused_recurrent`. With autograd, `chunk` is always used.
            Otherwise, inputs of at most 64 tokens in eval mode use `fused_recurrent`. Default: `chunk`.
        use_beta_gate (bool, Optional):
            Whether to scale keys and values by a learned input gate. Default: `True`.
        use_alpha_connection (bool, Optional):
            Whether to mix the ridge solution with the raw query through a learned gate; if `False`, the ridge
            solution is used directly. Default: `True`.
        use_v_conv (bool, Optional):
            Whether to apply a short convolution to the values. Default: `True`.
        use_forgetting_gate (bool, Optional):
            Whether to decay `U_t`. Default: `True`.
        use_forgetting_gate_kk (bool, Optional):
            Whether to decay `H_t` with the same gate; only used if `use_forgetting_gate` is `True`. Default: `True`.
        gla_rescale (bool, Optional):
            Whether to scale the readout by `1 / sqrt(head_dim)`. Default: `True`.
        ridge_ratio (float, Optional):
            Sets the ridge strength `lamb_t = ridge_ratio * ||H_t||_F`. Default: 0.02.
        num_iter (int, Optional):
            The number of Chebyshev iterations for the ridge solve, at least 1. Default: 30.
        use_gate (bool, Optional):
            Whether to use an output gate. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolutions. Default: 4.
        layer_idx (int, Optional):
            The index of the layer. Default: `None`.
        norm_eps (float, Optional):
            The epsilon value for the output normalization. Default: 1e-6.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        num_kv_heads: int | None = None,
        mode: str = 'chunk',
        use_beta_gate: bool = True,
        use_alpha_connection: bool = True,
        use_v_conv: bool = True,
        use_forgetting_gate: bool = True,
        use_forgetting_gate_kk: bool = True,
        gla_rescale: bool = True,
        ridge_ratio: float = 0.02,
        num_iter: int = 30,
        use_gate: bool = True,
        conv_size: int = 4,
        layer_idx: int | None = None,
        norm_eps: float = 1e-6,
        **kwargs,
    ) -> GatedKalmaNet:
        super().__init__()

        if mode not in ('chunk', 'fused_recurrent'):
            raise ValueError(f"Not supported mode `{mode}`.")
        if num_iter < 1:
            raise ValueError(f"`num_iter` must be at least 1, got {num_iter}.")
        if num_kv_heads is not None:
            if num_v_heads is not None:
                raise ValueError("`num_kv_heads` (GQA) and `num_v_heads` (GVA) cannot be set together.")
            if num_heads % num_kv_heads != 0:
                raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}.")
            num_k_heads = num_v_heads = num_kv_heads
        else:
            num_k_heads = num_heads
            num_v_heads = num_v_heads if num_v_heads is not None else num_heads
            if num_v_heads % num_heads != 0:
                raise ValueError(f"num_v_heads={num_v_heads} must be divisible by num_heads={num_heads}.")
        head_v_dim = int(head_dim * expand_v)
        if not math.isclose(head_dim * expand_v, head_v_dim, rel_tol=1e-5):
            raise ValueError(f"expand_v={expand_v} does not give an integer value head dim for head_dim={head_dim}.")

        self.hidden_size = hidden_size
        self.mode = mode
        self.head_dim = head_dim
        self.head_v_dim = head_v_dim
        self.num_heads = num_heads
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        # all heads are repeated to this count, since the gates are per expanded head
        self.num_expanded_heads = max(num_heads, num_v_heads)
        self.use_beta_gate = use_beta_gate
        self.use_alpha_connection = use_alpha_connection
        self.use_v_conv = use_v_conv
        self.use_forgetting_gate = use_forgetting_gate
        self.use_forgetting_gate_kk = use_forgetting_gate_kk
        self.scale = head_dim ** -0.5 if gla_rescale else 1.
        self.ridge_ratio = ridge_ratio
        self.num_iter = num_iter
        self.use_gate = use_gate
        self.conv_size = conv_size
        self.layer_idx = layer_idx

        H = self.num_expanded_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_k_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_v_heads * head_v_dim, bias=False)
        if use_beta_gate:
            self.b_proj = nn.Linear(hidden_size, H, bias=True)
        if use_alpha_connection:
            self.alpha_proj = nn.Linear(hidden_size, H, bias=True)

        if use_forgetting_gate:
            self.a_proj = nn.Linear(hidden_size, H, bias=False)
            A = torch.empty(H, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True
            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            dt = torch.exp(torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt = torch.clamp(dt, min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True

        self.q_conv1d = ShortConvolution(hidden_size=num_heads * head_dim, kernel_size=conv_size, activation='silu')
        self.k_conv1d = ShortConvolution(hidden_size=num_k_heads * head_dim, kernel_size=conv_size, activation='silu')
        if use_v_conv:
            self.v_conv1d = ShortConvolution(
                hidden_size=num_v_heads * head_v_dim,
                kernel_size=conv_size,
                activation='silu',
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, H * head_v_dim, bias=False)
            self.o_norm = FusedRMSNormGated(head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(H * head_v_dim, hidden_size, bias=False)

    def _expand(self, x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        x = rearrange(x, '... (h d) -> ... h d', h=num_heads, d=head_dim)
        if num_heads < self.num_expanded_heads:
            x = repeat(x, '... h d -> ... (h g) d', g=self.num_expanded_heads // num_heads)
        return x

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
        if torch.is_grad_enabled():
            mode = 'chunk'
        elif q_len <= 64 and not self.training:
            mode = 'fused_recurrent'
        else:
            mode = self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get('cu_seqlens')
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, cu_seqlens, attention_mask, q_len)

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
        if self.use_v_conv:
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            v = self.v_proj(hidden_states)

        q = l2_norm(self._expand(q, self.num_heads, self.head_dim))
        k = l2_norm(self._expand(k, self.num_k_heads, self.head_dim))
        v = self._expand(v, self.num_v_heads, self.head_v_dim)
        if self.use_beta_gate:
            beta = self.b_proj(hidden_states).sigmoid()[..., None] + 1e-6
            k, v = (beta * k).to(k.dtype), (beta * v).to(v.dtype)
        alpha = self.alpha_proj(hidden_states).sigmoid() if self.use_alpha_connection else None
        g, gk = None, None
        if self.use_forgetting_gate:
            g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
            gk = g if self.use_forgetting_gate_kk else None

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        gka = chunk_gka if mode == 'chunk' else fused_recurrent_gka
        o, recurrent_state = gka(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            alpha=alpha,
            scale=self.scale,
            ridge_ratio=self.ridge_ratio,
            num_iter=self.num_iter,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )

        if self.use_gate:
            gate = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, gate)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        o = repad_hidden_states(o, indices, batch_size, q_len)

        return o, None, past_key_values
