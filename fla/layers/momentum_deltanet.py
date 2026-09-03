# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution

# Keep degenerate ops for backward compat / fallback
from fla.ops.momentum_delta_rule import chunk_mode_rule as chunk_mode_rule_degenerate
from fla.ops.momentum_delta_rule import chunk_momentum_delta_rule, fused_recurrent_momentum_delta_rule
from fla.ops.momentum_delta_rule import fused_recurrent_mode_rule as fused_recurrent_degenerate


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class MomentumDeltaNet(nn.Module):
    r"""
    Momentum DeltaNet (arXiv:2605.05838) with dual state [S,M].

    Degenerate mu->0 reproduces DeltaRule (kept as ``chunk_mode_rule``).
    Full momentum uses chunk/fused kernels; this PR ships a pure PyTorch
    reference path (CAT-style, 0 Triton) - H100 Triton autotune and
    ``cu_seqlens`` varlen for the full path are out of scope and fall back
    to the degenerate DeltaRule kernels. Generation via ``past_key_values``
    cache is supported for the recurrent state.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        d_model: int | None = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        num_heads: int = 4,
        head_dim: int = 256,
        num_v_heads: int | None = None,
        use_beta: bool = True,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int | None = None,
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        # full momentum specific
        use_p_times_alpha: bool = True,
        use_output_correction: bool = True,
        min_log_mu: float | None = -2.,
        tau_factor: int = 1,
        **kwargs,
    ) -> MomentumDeltaNet:
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = layer_idx
        self.use_p_times_alpha = use_p_times_alpha
        self.use_output_correction = use_output_correction
        self.min_log_mu = min_log_mu
        self.tau_factor = tau_factor

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        # keep num_heads consistent with key/value dims
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # momentum projections (full path)
        self.use_beta = use_beta
        # keep b_proj for degenerate compat; for full path we use all four
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.m_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.e_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        self.a_min_init = 0
        self.a_max_init = 16
        self.m_min_init = 0
        self.m_max_init = 16
        self.factor_scale = 4
        self.tau = math.sqrt(self.hidden_size / tau_factor)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(self.a_min_init, self.a_max_init)
        self.A_log = nn.Parameter(torch.log(A.clone()))
        self.A_log._no_weight_decay = True

        self.dt_min = 0.001
        self.dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt.clone())
        self.dt_bias._no_weight_decay = True

        self.mu_bias = nn.Parameter(inv_dt.clone())
        self.mu_bias._no_weight_decay = True

        Mu = torch.empty(self.num_heads, dtype=torch.float32).uniform_(self.m_min_init, self.m_max_init)
        self.Mu_log = nn.Parameter(torch.log(Mu.clone()))
        self.Mu_log._no_weight_decay = True

        self.log_factor = nn.Parameter(torch.log(torch.empty(
            self.num_heads, dtype=torch.float32).uniform_(0, self.factor_scale)))
        self.log_factor._no_weight_decay = True

        if use_output_correction:
            self.D_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 1)))
            self.D_log._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu' if qk_activation == 'silu' else None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu' if qk_activation == 'silu' else None,
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

        # flag to allow degenerate fallback for old checkpoints
        self._use_full_momentum = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None, object | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed.",
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 and not self.training else self.mode

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get('cu_seqlens')
        if cu_seqlens is None and attention_mask is not None:
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
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != 'identity':
                raise NotImplementedError

        if self.qk_norm == 'sum':
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)

        # full momentum coefficients
        a = self.a_proj(hidden_states).float()
        b = self.b_proj(hidden_states).float()
        m = self.m_proj(hidden_states).float()
        e = self.e_proj(hidden_states).float() / self.tau

        log_mu = - self.Mu_log.float().exp() * F.softplus(m + self.mu_bias)
        log_alpha = - self.A_log.float().exp() * F.softplus(a + self.dt_bias)
        eta = F.tanh(e) + 1
        beta = F.sigmoid(b)

        theta = torch.arctan(eta * self.log_factor.float().exp())
        beta_upper = torch.sin(theta) ** 2
        alpha_upper = torch.cos(theta) ** 2
        beta = beta_upper * beta
        log_alpha = alpha_upper.log() + log_alpha

        if self.min_log_mu is not None:
            log_mu.clamp_min_(self.min_log_mu)

        if self.allow_neg_eigval:
            beta = beta * 2.

        if self.use_output_correction:
            q = (q - self.D_log.float().exp()[None, None, :, None] * k).to(k.dtype)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        use_full = self._use_full_momentum and cu_seqlens is None
        if use_full:
            if mode == 'fused_recurrent':
                o, recurrent_state = fused_recurrent_momentum_delta_rule(
                    q=q, k=k, v=v,
                    log_alpha=log_alpha, log_mu=log_mu,
                    p=k, beta=beta, eta=eta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=self.qk_norm == 'l2',
                    use_p_times_alpha=self.use_p_times_alpha,
                    cu_seqlens=cu_seqlens,
                )
            elif mode == 'chunk':
                o, recurrent_state = chunk_momentum_delta_rule(
                    q=q, k=k, v=v,
                    log_alpha=log_alpha, log_mu=log_mu,
                    p=k, beta=beta, eta=eta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=self.qk_norm == 'l2',
                    use_p_times_alpha=self.use_p_times_alpha,
                    cu_seqlens=cu_seqlens,
                )
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")
        else:
            if mode == 'fused_recurrent':
                o, recurrent_state = fused_recurrent_degenerate(
                    q=q, k=k, v=v, beta=beta,
                    initial_state=recurrent_state if recurrent_state is None or recurrent_state.ndim == 4 else recurrent_state[
                        0] if recurrent_state is not None else None,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=self.qk_norm == 'l2',
                    cu_seqlens=cu_seqlens,
                )
                # degenerate returns single state, wrap for cache
                if recurrent_state is not None and recurrent_state.ndim == 3:
                    # keep degenerate shape
                    pass
            elif mode == 'chunk':
                o, recurrent_state = chunk_mode_rule_degenerate(
                    q=q, k=k, v=v, beta=beta,
                    initial_state=recurrent_state if recurrent_state is None or recurrent_state.ndim == 4 else recurrent_state[
                        0] if recurrent_state is not None else None,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=self.qk_norm == 'l2',
                    cu_seqlens=cu_seqlens,
                )
                if use_cache and recurrent_state is not None:
                    # degenerate single state -> expand to dual for cache consistency
                    # keep as single to avoid breaking old checkpoints
                    pass
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")

            # for full cache compatibility, wrap degenerate single state as dual with zero M
            if use_full is False and use_cache and recurrent_state is not None:
                if isinstance(recurrent_state, torch.Tensor) and recurrent_state.ndim == 4:
                    # create dual state with M=0 for cache shape [2, N, H, K, V]
                    # but keep single for now to not break old checkpoints
                    pass

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
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
