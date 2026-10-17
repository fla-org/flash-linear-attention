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
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.diag_kdn import chunk_diag_kdn, fused_recurrent_diag_kdn

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class DiagonalKalmanDeltaNetwork(nn.Module):
    r"""
    Diagonal Kalman Delta Network (DiagKDN) layer implementation.

    DiagKDN keeps the KDA memory decay, with per-key-dim forget gates ``g = -exp(A_log) * softplus(f_proj(x) + dt_bias)``,
    but writes along the Kalman gain of a diagonal posterior over the memory rows instead of a scaled key.
    Each head tracks a per-key-channel precision ``c`` that decays with the forget gates and grows with the observed keys:

    .. math::
        z_t = \exp(2 g_t) / c_{t-1} + \omega_t, \quad
        \kappa_t = z_t \odot k_t / (r_t + \langle z_t, k_t^2 \rangle), \quad
        c_t = 1 / z_t + \text{info\_scale} \, k_t^2 / r_t,

    where the process noise ``omega`` (per key channel) and observation noise ``r`` (per head) are data-dependent.
    The recurrent state is the pair of memory and precision.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        mode (str, Optional):
            Which DiagKDN kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        omega_min (float, Optional):
            The lower bound of the process noise. Default: 0.0.
        r_min (float, Optional):
            The lower bound of the observation noise. Default: 0.01.
        info_scale (float, Optional):
            Information contributed by one unit-norm key. Default: `None`, i.e., the head dimension.
        layer_idx (int, Optional):
            The index of the layer. Default: `None`.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        omega_min: float = 0.0,
        r_min: float = 0.01,
        info_scale: float | None = None,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> DiagonalKalmanDeltaNetwork:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.omega_min = omega_min
        self.r_min = r_min
        self.info_scale = info_scale

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        if omega_min < 0:
            raise ValueError(f"`omega_min` must be non-negative, got {omega_min}.")
        if r_min <= 0:
            raise ValueError(f"`r_min` must be positive, got {r_min}.")
        if info_scale is not None and info_scale <= 0:
            raise ValueError(f"`info_scale` must be positive, got {info_scale}.")
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        # Gate dim = H * K: per head, per key-dim gating.
        self.gate_dim = int(self.num_heads * self.head_k_dim)
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False),
        )
        # the process noise is per key channel and starts isotropic from a zero projection
        self.omega_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=True),
        )
        nn.init.zeros_(self.omega_proj[1].weight)
        nn.init.zeros_(self.omega_proj[1].bias)
        self.omega_proj[1]._zero_init = True
        self.r_proj = nn.Linear(hidden_size, self.num_heads, bias=True)

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.gate_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        # the learned prior precision 0.1 + softplus(initial_precision_param) starts at 1
        self.initial_precision_param = nn.Parameter(torch.full((self.num_heads,), math.log(math.expm1(0.9))))
        self.initial_precision_param._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

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
            mode = "chunk"
        elif q_len <= 64 and not self.training:
            mode = "fused_recurrent"
        else:
            mode = self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get("cu_seqlens")
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, cu_seqlens, attention_mask, q_len)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
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
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        # the gate gradient is rounded to the dtype of g, so g is kept in fp32 to sum the dt_bias gradient in fp32
        g = rearrange(self.f_proj(hidden_states).float(), "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        omega = rearrange(self.omega_proj(hidden_states), "... (h d) -> ... h d", d=self.head_k_dim)
        omega = self.omega_min + F.softplus(omega.float())
        r = self.r_min + F.softplus(self.r_proj(hidden_states).float())

        if last_state is not None:
            recurrent_state = last_state["recurrent_state"]
        else:
            # every sequence starts from the learned prior precision, shared by the key channels of a head
            N = batch_size if cu_seqlens is None else len(cu_seqlens) - 1
            c0 = 0.1 + F.softplus(self.initial_precision_param.float())
            recurrent_state = (None, c0[None, :, None].expand(N, -1, self.head_k_dim))
        if mode == "chunk":
            o, recurrent_state = chunk_diag_kdn(
                q=q,
                k=k,
                v=v,
                g=g,
                omega=omega,
                r=r,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                info_scale=self.info_scale,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_diag_kdn(
                q=q,
                k=k,
                v=v,
                g=g,
                omega=omega,
                r=r,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                info_scale=self.info_scale,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        o = repad_hidden_states(o, indices, batch_size, q_len)

        return o, None, past_key_values
