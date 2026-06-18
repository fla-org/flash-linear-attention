# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.precond_gated_delta_rule import (
    chunk_precond_gated_delta_rule,
    fused_recurrent_precond_gated_delta_rule,
    naive_recurrent_precond_gated_delta_rule,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class PrecondGatedDeltaNet(nn.Module):
    """
    Preconditioned Gated Delta Networks (PGDN) layer implementation.

    Reference: `Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences <https://arxiv.org/abs/2604.21100>`_

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.

    Parameter allocation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 6.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA (Grouped Value Attention) is applied if `num_v_heads` > `num_heads`,
            where `num_v_heads` must be divisible by `num_heads`.
            The kernels natively support GVA by mapping multiple value heads to each query/key head.
            Default: `None`.
        mode (str, Optional):
            Which Preconditioned Gated DeltaNet kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `naive`.
            Default: `chunk`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference:
            `Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues <https://arxiv.org/abs/2411.12537>`_
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
        squash_x (float, Optional):
            The squash range parameter. Squashes preconditioner onto interval [1/x, x].
            Needs to be <= 2.0 for eigenstability. Default: 1.5.
        squash_eps (float, Optional):
            The epsilon for numerical stability in the squash function. Default: 1e-6.
        log_atk_scale_init (float, Optional):
            The initial value for the learnable per-head log_atk_scale parameter. Default: -0.2.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        squash_x: float = 1.5,
        squash_eps: float = 1e-6,
        log_atk_scale_init: float = -0.2,
        **kwargs,
    ) -> PrecondGatedDeltaNet:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        self.squash_x = squash_x
        self.squash_eps = squash_eps
        self.log_atk_scale_init = log_atk_scale_init

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ['chunk', 'fused_recurrent', 'naive'], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Gate and beta projections for preconditioned Gated DeltaNet recurrence
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        # Separate gate and beta projections for ATK preconditioner recurrence
        self.a_atk_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_atk_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        # Learnable per-head log-space center for preconditioner
        self.log_atk_scale = nn.Parameter(
            torch.full((self.num_v_heads,), self.log_atk_scale_init, dtype=torch.float32)
        )
        self.log_atk_scale._no_weight_decay = True

        # Gate parameters for preconditioned Gated DeltaNet main recurrence
        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded (taken from Gated DeltaNet layer)
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # Gate parameters for preconditioner recurrence
        A_atk = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16)
        self.A_log_atk = nn.Parameter(torch.log(A_atk))
        self.A_log_atk._no_weight_decay = True

        dt_atk = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt_atk = torch.clamp(dt_atk, min=dt_init_floor)
        inv_dt_atk = dt_atk + torch.log(-torch.expm1(-dt_atk))
        self.dt_bias_atk = nn.Parameter(inv_dt_atk)
        self.dt_bias_atk._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing.",
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
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

        # change to inference mode.
        mode = self.mode
        if q_len <= 64 and not self.training:
            mode = 'fused_recurrent'
        if self.training:
            assert mode in ['chunk', 'naive'], "Only chunk/naive modes are supported in training."

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
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

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        beta_atk = self.b_atk_proj(hidden_states).sigmoid()
        g_atk = -self.A_log_atk.float().exp() * F.softplus(self.a_atk_proj(hidden_states).float() + self.dt_bias_atk)
        log_atk_scale = self.log_atk_scale

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        A_state = last_state['A_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state, A_state = chunk_precond_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                g_atk=g_atk,
                beta=beta,
                beta_atk=beta_atk,
                initial_state=recurrent_state,
                initial_A_state=A_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                x=self.squash_x,
                eps=self.squash_eps,
                log_atk_scale=log_atk_scale,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state, A_state = fused_recurrent_precond_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                g_atk=g_atk,
                beta=beta,
                beta_atk=beta_atk,
                initial_state=recurrent_state,
                initial_A_state=A_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                x=self.squash_x,
                eps=self.squash_eps,
                log_atk_scale=log_atk_scale,
            )
        elif mode == 'naive':
            o, recurrent_state, A_state = naive_recurrent_precond_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                g_atk=g_atk,
                beta=beta,
                beta_atk=beta_atk,
                initial_state=recurrent_state,
                initial_A_state=A_state,
                output_final_state=use_cache,
                x=self.squash_x,
                eps=self.squash_eps,
                log_atk_scale=log_atk_scale,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            A_state=A_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
