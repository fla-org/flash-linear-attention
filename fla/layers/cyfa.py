# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# Portions adapted from CyclicFlowAttention, Copyright (c) 2026 Yixiao Chen.
# https://github.com/Chyxx/CyclicFlowAttention

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.cyfa.chunk import chunk_cyfa
from fla.ops.cyfa.fused_recurrent import fused_recurrent_cyfa

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class CyclicFlowAttention(nn.Module):
    """
    Cyclic Flow Attention (CyFA) layer implementation.

    Reference: `CyFA: Linear Sequence Modeling with Relative-Time-Partitioned Memory
    <https://arxiv.org/abs/2609.36259>`_

    CyFA maintains aligned key and value memories over relative-time slots. A learned clock transports stored
    associations through the slots, while content-based readout retrieves from the resulting memory.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dimension. Default: 1.0.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        mode (str, Optional):
            Which CyFA kernel to use.
            Currently available: ``chunk`` and ``fused_recurrent``.
            Default: ``chunk``.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: ``True``.
        conv_size (int, Optional):
            The kernel size of the short convolution. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution. Default: ``False``.
        num_slots (int, Optional):
            The number of rows used to store the relative-time memory. Default: 128.
        checkpoint_level (int, Optional):
            The activation checkpointing level used by the chunk kernel. Default: 0.
        layer_idx (int, Optional):
            The index of the layer. Default: ``None``.
        norm_eps (float, Optional):
            The epsilon value for the normalization layers. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        head_dim: int = 256,
        num_heads: int = 4,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_slots: int = 128,
        checkpoint_level: int = 0,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.head_k_dim = int(self.head_dim * self.expand_k)
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        if not math.isclose(self.head_dim * self.expand_k, self.head_k_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_k={expand_k} does not produce an integer head dimension for head_dim={head_dim}.",
            )
        if not math.isclose(self.head_dim * self.expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer head dimension for head_dim={head_dim}.",
            )
        self.num_slots = int(num_slots)
        if self.num_slots <= 0 or self.num_slots % 2:
            raise ValueError("`num_slots` must be a positive even integer.")
        if checkpoint_level not in (0, 1):
            raise ValueError("`checkpoint_level` must be either 0 or 1.")
        assert mode in ("chunk", "fused_recurrent"), f"Not supported mode `{mode}`."
        self.num_readout_slots = self.num_slots - 1
        self.checkpoint_level = checkpoint_level

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        dt = torch.exp(
            torch.rand(self.num_heads, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001),
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.d_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(0.0, 16.0)),
        )
        self.A_log._no_weight_decay = True
        self.readout = nn.Parameter(
            torch.eye(self.num_readout_slots, dtype=torch.float32).expand(self.num_heads, -1, -1).clone(),
        )
        self.q_norm = RMSNorm(self.head_k_dim, eps=norm_eps, dtype=torch.float32)
        self.k_norm = RMSNorm(self.head_k_dim, eps=norm_eps, dtype=torch.float32)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )

        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=norm_eps,
            activation="sigmoid",
        )
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if any(isinstance(module.weight, DTensor) for module in (self.q_proj, self.k_proj, self.v_proj)):
            return self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        qkv = F.linear(
            hidden_states,
            torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0),
        )
        return torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

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

        indices = None
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is None and attention_mask is not None and q_len > 1:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q_proj, k_proj, v_proj = self._project_qkv(hidden_states)
            conv_output_final_state = bool(use_cache or (q_len == 1 and last_state is None))
            q, conv_state_q = self.q_conv1d(
                x=q_proj,
                cache=conv_state_q,
                output_final_state=conv_output_final_state,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=k_proj,
                cache=conv_state_k,
                output_final_state=conv_output_final_state,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=v_proj,
                cache=conv_state_v,
                output_final_state=conv_output_final_state,
                cu_seqlens=cu_seqlens,
            )
        else:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            q, k, v = self._project_qkv(hidden_states)
        a = self.a_proj(hidden_states)
        beta = self.b_proj(hidden_states).float().sigmoid()
        delta = self.d_proj(hidden_states.detach()).float().sigmoid()

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        decay = F.softplus(a.float() + self.dt_bias.float())
        g = -self.A_log.float().exp() * decay
        if mode == "chunk":
            o, recurrent_state = chunk_cyfa(
                q=q,
                k=k,
                v=v,
                g=g,
                delta=delta,
                beta=beta,
                q_norm_weight=self.q_norm.weight,
                k_norm_weight=self.k_norm.weight,
                q_norm_eps=self.q_norm.eps,
                k_norm_eps=self.k_norm.eps,
                readout=self.readout,
                scale=self.head_k_dim**-0.5,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                checkpoint_level=self.checkpoint_level,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_cyfa(
                q=q,
                k=k,
                v=v,
                g=g,
                delta=delta,
                beta=beta,
                readout=self.readout,
                q_norm_weight=self.q_norm.weight,
                k_norm_weight=self.k_norm.weight,
                q_norm_eps=self.q_norm.eps,
                k_norm_eps=self.k_norm.eps,
                scale=self.head_k_dim**-0.5,
                initial_state=recurrent_state,
                output_final_state=use_cache,
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

        # keep small readout activations above the RMSNorm epsilon floor.
        o = self.o_norm(
            o * self.num_readout_slots,
            rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim),
        )
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if indices is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
