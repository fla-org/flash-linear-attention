# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).


"""
Switching Linear Attention Layer

A linear attention mechanism with switching capabilities based on
a recurrent mixture of experts formulation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules.swila import initialization
from fla.modules.swila.argument_checks import check_recurrent_mode, validate_swila_args
from fla.modules.swila.diagnostics import active_swila_capture
from fla.modules.swila.packing import (
    repad_packed_tensor,
    validate_packed_boundaries,
    validate_sequence_state_counts,
)
from fla.modules.swila.projections import LowRankProjection, SwiGLUProjection
from fla.modules.swila.router import compute_router_logits, compute_temporal_gates
from fla.ops.swila import compute_swila_mi_loss

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache

from fla.layers.utils import (
    get_layer_cache,
    pad_input,
    repad_hidden_states,
    unpad_hidden_states,
    update_layer_cache,
)
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution


class SwitchingLinearAttention(nn.Module):
    """
    Switching Linear Attention layer.

    A linear attention mechanism with switching capabilities based on
    a factorial mixture of linear regression formulation with fused recurrent
    computation.

    Args:
        mode (str): Operation mode for SwiLA computation.
            - 'fused_recurrent': Fused Triton kernel for recurrent computation (default).
            The PyTorch reference is available separately from
            fla.ops.swila.naive.swila_recurrence, not as a layer mode.
        d_model (int): Input/output dimensionality. Required.
        expand_k (float): Expansion factor for key dimension. key_dim = d_model * expand_k.
        expand_v (float): Expansion factor for value dimension. value_dim = d_model * expand_v.
        num_heads (int): Number of attention heads.
        num_experts (int): Number of mixture components/experts (K).
        architecture_profile (str): 'current' keeps value-shared beta and
            calibrated initialization. Explicit 'historical_8dc05f93' restores
            expert-specific SwiGLU beta and independent Kaiming router/beta
            initialization for the historical plain-SwiLA architecture.
        use_gate (bool): Whether to use output gating (adds g_proj linear layer).
        use_short_conv (bool): Whether to use FLA ShortConvolution on q, k, v after projections.
        conv_size (int): Kernel size for convolutions.
        conv_bias (bool): Whether to use bias in convolutions.
        layer_idx (int): Layer index for caching.
        qk_norm (str | None): Normalization for Q/K ('l2' or None).
            Signed-sum normalization ('sum') is no longer supported.
        norm_eps (float): Epsilon for layer normalization. Default: 1e-5.
        scale (float): Scale factor for output. If None, defaults to 1/sqrt(D_k).
        return_output_only (bool): If True, forward() returns only the output tensor instead of
            a tuple (output, attentions, past_key_values, aux_loss). Default: True.
        use_pi_conv (bool): Whether to apply short convolution to pi-path
            features before the terminal logit projection. Default: False.
        pi_proj_rank (int): Low-rank dimension for pi projections. When >0,
            pi uses a global projection from pi_input_dim through this rank to
            [H, D_v, E]. When 0, one shared per-head Linear maps head_pi_dim to
            [D_v, E]. A narrow linear path whose input width already equals the
            rank collapses to one equivalent global Linear. Default: 256.
        b_proj_rank (int): Low-rank dimension for beta (learning rate) projection. When >0, uses a
            low-rank projection (d_model -> rank -> beta_dim).
            When 0, uses full-rank nn.Linear(d_model, beta_dim). Default: 16.
        low_rank_proj_type (str): Shared fallback projection type for pi when
            pi_low_rank_proj_type is unset. 'linear': LowRankProjection
            (up(down(x))). 'swiglu': SwiGLUProjection
            (out(silu(gate(x)) * feat(x))). Default: 'swiglu'.
        pi_low_rank_proj_type (str | None): Optional projection type override for
            pi_q/pi_k projections. Defaults to low_rank_proj_type.
        pi_init_logit_std (float): Approximate target standard deviation of raw
            pi logits for scaled normal initialization without a pi convolution.
            Default: 0.1.
        pi_align_qk_init (bool): Whether pi_q and pi_k start from identical but
            separately owned input projections, optional pi convolutions, and
            terminal projections. Independent injected noise can still make
            their sampled noisy logits differ. Default: True.
        pi_noise_init_std (float): Initial standard deviation of injected pi
            router noise when inject_noise=True. Zero noise-projection weights
            and an inverse-softplus bias make this scale input-independent at
            initialization. Default: 0.02.
        b_low_rank_proj_type (str | None): Optional projection type override for
            beta projections. Defaults to 'linear'.
        alpha_low_rank_proj_type (str | None): Optional projection type override
            for forget-gate alpha projections. Defaults to 'linear'.

        init_state_mode (str): Initialization mode for recurrent state.
            - 'zeros': Fixed zero initial state.
            - 'learnable': Zero-initialized trainable parameter [1, H, D_v, K, D_k] (default).
        temporal (bool): Whether to use temporal filtering on expert assignments.
            Uses exact identity transitions with reset gates.

        inject_noise (bool): Whether to inject Gaussian noise into pi logits
            during training. Its learned standard deviation starts at
            pi_noise_init_std. Default: False.
        inject_gate_noise (bool): Whether to inject noise into gate values (reset_gate mode).

        gate_init_bias (float): Initial temporal reset-gate logit bias.
            sigmoid(bias) weights the fresh pi prior and 1-sigmoid(bias)
            weights the previous responsibility. Default: -3.0.
        gate_noise_init_std (float): Initial standard deviation for optional
            temporal reset-gate logit noise. Default: 0.02.
        mi_loss_weight (float): Weight for mutual information auxiliary loss. Default: 0.0 (disabled).
        use_forget_gate (bool): Whether to enable Mamba-style input-dependent
            state decay. Default: False.
        alpha_proj_rank (int): Low-rank dimension for the alpha projection.
            When 0, use a direct Linear. Default: 16.

    Note:
        Routers pass raw expert logits; recurrence paths normalize them with
        softmax where responsibilities are needed.
        Current-profile beta is per value channel and shared over experts;
        historical-profile beta has a separate value-channel rate per expert.
        State decay is
        per key channel, shared over values and experts; both routing and
        writing use the residual computed from the decayed state.

    Initialization:
        reset_parameters() restores the router, beta, decay and temporal-gate
        initialization after generic model initialization. Implementation and
        optimizer metadata live in fla.modules.swila.initialization; optional
        verification lives in fla.modules.swila.initialization_checks.
        The historical profile does not perform an extra constructor reset:
        child constructors initialize weights, then the owning model controls
        the original reset/RNG sequence. Its resets do not center or align
        routers and do not apply the current profile's calibrated scales.
    """

    # Generic initializers visit Linear children before this composite module.
    # Backbones can use this marker to restore SwiLA-specific initialization after
    # their generic leaf initialization without importing this class eagerly.
    requires_swila_reset_after_generic_init = True

    @validate_swila_args
    def __init__(
        self,
        mode: Literal["fused_recurrent"] = 'fused_recurrent',
        d_model: int = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_experts: int = 4,
        use_gate: bool = False,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        qk_norm: Literal["l2"] | None = 'l2',
        norm_eps: float = 1e-5,
        scale: float = None,
        return_output_only: bool = True,
        use_pi_conv: bool = False,
        pi_proj_rank: int = 256,
        b_proj_rank: int = 16,
        low_rank_proj_type: str = 'swiglu',
        pi_low_rank_proj_type: str | None = None,
        pi_init_logit_std: float = 0.1,
        pi_align_qk_init: bool = True,
        pi_noise_init_std: float = 0.02,
        b_low_rank_proj_type: str | None = 'linear',
        alpha_low_rank_proj_type: str | None = 'linear',
        init_state_mode: str = 'learnable',
        temporal: bool = False,
        inject_noise: bool = False,
        inject_gate_noise: bool = False,
        # Expert auxiliary loss parameters

        gate_init_bias: float = -3.0,
        gate_noise_init_std: float = 0.02,
        mi_loss_weight: float = 0.0,
        use_forget_gate: bool = False,
        alpha_proj_rank: int = 16,
        *,
        architecture_profile: Literal["current", "historical_8dc05f93"] = 'current',
    ) -> SwitchingLinearAttention:
        super().__init__()

        self.mode = mode
        self.architecture_profile = architecture_profile
        self.qk_norm = qk_norm
        self.scale = scale  # If None, recurrence defaults to 1/sqrt(D_k).
        self.return_output_only = return_output_only
        self.use_pi_conv = use_pi_conv
        self.pi_proj_rank = pi_proj_rank
        self.b_proj_rank = b_proj_rank

        self.low_rank_proj_type = 'swiglu' if low_rank_proj_type in (None, 'null') else low_rank_proj_type
        self.pi_low_rank_proj_type = (
            self.low_rank_proj_type if pi_low_rank_proj_type in (None, 'null') else pi_low_rank_proj_type
        )
        self.pi_init_logit_std = pi_init_logit_std
        self.pi_align_qk_init = pi_align_qk_init
        self.pi_noise_init_std = pi_noise_init_std
        self.b_low_rank_proj_type = 'linear' if b_low_rank_proj_type in (None, 'null') else b_low_rank_proj_type
        self.alpha_low_rank_proj_type = 'linear' if alpha_low_rank_proj_type in (None, 'null') else alpha_low_rank_proj_type

        self.init_state_mode = init_state_mode
        self.temporal = temporal

        self.inject_noise = inject_noise
        self.inject_gate_noise = inject_gate_noise

        self.gate_init_bias = gate_init_bias
        self.gate_noise_init_std = gate_noise_init_std
        self.mi_loss_weight = mi_loss_weight
        self.use_forget_gate = use_forget_gate
        self.alpha_proj_rank = alpha_proj_rank

        # Auxiliary loss tracking (populated during forward, used by return_output_only path)
        self._aux_loss = None

        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx

        # Learnable initial state: [1, H, D_v, K, D_k] (broadcasts across batch)
        if self.init_state_mode == 'learnable':
            init_state_shape = (1, self.num_heads, self.head_v_dim, self.num_experts, self.head_k_dim)
            self.init_state = nn.Parameter(torch.zeros(*init_state_shape))
        else:
            self.init_state = None

        # Shared router dimensions for terminal projections, noise and gates.
        pi_input_dim = pi_proj_rank if pi_proj_rank > 0 else self.key_dim
        head_pi_dim = pi_input_dim // self.num_heads

        # Gate projections for temporal reset gates
        if self.temporal:
            gate_out_dim = self.head_v_dim
            self.gate_kv_proj = nn.Linear(head_pi_dim, gate_out_dim, bias=True)
            self.gate_q_proj = nn.Linear(head_pi_dim, gate_out_dim, bias=True)
            # Biases encode the baseline temporal time constants and must not
            # drift toward sigmoid(0)=0.5 solely because of AdamW decay.
            self.gate_kv_proj.bias._no_weight_decay = True
            self.gate_q_proj.bias._no_weight_decay = True
            # Noise injection for gate logits (same pattern as pi noise)
            if self.inject_gate_noise:
                self.W_noise_gate_kv = nn.Linear(
                    head_pi_dim,
                    gate_out_dim,
                    bias=True,
                )
                self.W_noise_gate_q = nn.Linear(
                    head_pi_dim,
                    gate_out_dim,
                    bias=True,
                )
                # These biases encode softplus-domain noise standard deviations.
                self.W_noise_gate_kv.bias._no_weight_decay = True
                self.W_noise_gate_q.bias._no_weight_decay = True
        else:
            self.gate_kv_proj = None
            self.gate_q_proj = None
            self.W_noise_gate_kv = None
            self.W_noise_gate_q = None

        # Noise injection for load balancing (Noisy Top-K Gating style).
        # Bias initialization makes Softplus(W_noise(pi_input)) start at the
        # configured pi_noise_init_std instead of Softplus(0) ~= 0.693.
        if self.inject_noise:
            # W_noise projects pi_input [H, head_pi_dim] to noise magnitude
            noise_out_dim = self.head_v_dim * self.num_experts
            self.W_noise_k = nn.Linear(head_pi_dim, noise_out_dim, bias=True)
            self.W_noise_q = nn.Linear(head_pi_dim, noise_out_dim, bias=True)
        else:
            self.W_noise_k = None
            self.W_noise_q = None

        # Projections
        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)

        # The historical plain model owns a distinct beta for every expert.
        beta_proj_dim = self.value_dim * (num_experts if architecture_profile == 'historical_8dc05f93' else 1)

        def projection_cls(proj_type: str):
            return SwiGLUProjection if proj_type == 'swiglu' else LowRankProjection

        b_proj_cls = projection_cls(self.b_low_rank_proj_type)
        alpha_proj_cls = projection_cls(self.alpha_low_rank_proj_type)
        pi_proj_cls = projection_cls(self.pi_low_rank_proj_type)
        if b_proj_rank > 0:
            self.b_proj = b_proj_cls(d_model, beta_proj_dim, rank=b_proj_rank)
        else:
            self.b_proj = nn.Linear(d_model, beta_proj_dim, bias=False)

        # Forget gate (alpha) parameters — Mamba-style input-dependent decay
        if use_forget_gate:
            gate_dim = self.key_dim
            if alpha_proj_rank > 0:
                self.a_proj = alpha_proj_cls(d_model, gate_dim, rank=alpha_proj_rank)
            else:
                self.a_proj = nn.Linear(d_model, gate_dim, bias=False)

            # One base decay magnitude per head. exp(A_log) broadcasts over
            # key channels, shared over experts and value channels.
            self.A_log = nn.Parameter(
                torch.zeros(self.num_heads, dtype=torch.float32)
            )
            self.A_log._no_weight_decay = True

            dt_shape = (self.num_heads, self.head_k_dim)
            self.dt_bias = nn.Parameter(torch.zeros(dt_shape, dtype=torch.float32))
            self.dt_bias._no_weight_decay = True

        # Expert-prior router. Terminal rows are later scaled and centered
        # over E so the initial prior is nearly uniform but not expert-identical.
        pi_out_dim = self.head_v_dim * self.num_experts

        # Routers project the input independently of Q/K/V.
        self.pi_input_dim = pi_input_dim
        self.head_pi_dim = head_pi_dim
        self.pi_q_input_proj = nn.Linear(d_model, pi_input_dim, bias=False)
        self.pi_k_input_proj = nn.Linear(d_model, pi_input_dim, bias=False)

        if pi_proj_rank > 0:
            # If the linear bottleneck has the input width, the two linear
            # maps collapse to one equivalent map.
            pi_proj_out = self.num_heads * pi_out_dim
            if self.pi_low_rank_proj_type == 'linear':
                def make_pi_proj(d_in: int, d_out: int, rank: int) -> nn.Module:
                    return nn.Linear(d_in, d_out, bias=False)
            else:
                make_pi_proj = pi_proj_cls
            self.pi_q_proj = make_pi_proj(pi_input_dim, pi_proj_out, rank=pi_proj_rank)
            self.pi_k_proj = make_pi_proj(pi_input_dim, pi_proj_out, rank=pi_proj_rank)
        else:
            self.pi_q_proj = nn.Linear(self.head_pi_dim, pi_out_dim, bias=False)
            self.pi_k_proj = nn.Linear(self.head_pi_dim, pi_out_dim, bias=False)

        # FLA ShortConvolution: applied to q, k, v after projections
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation="silu")
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation="silu")
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, bias=conv_bias, activation="silu")
            # Conv1d for pi (prior responsibility) pathway
            if use_pi_conv:
                pi_conv_dim = self.pi_input_dim
                self.pi_q_conv1d = ShortConvolution(pi_conv_dim, conv_size, bias=conv_bias, activation="silu")
                self.pi_k_conv1d = ShortConvolution(pi_conv_dim, conv_size, bias=conv_bias, activation="silu")
            else:
                self.pi_q_conv1d = None
                self.pi_k_conv1d = None
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off unless you know what you are doing.",
            )
            self.pi_q_conv1d = None
            self.pi_k_conv1d = None

        # Output normalization and projection
        if use_gate:
            self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.register_load_state_dict_pre_hook(
            initialization.capture_load_state_dict_prefix
        )
        self.register_load_state_dict_post_hook(
            initialization.restore_parameter_metadata_after_load
        )
        # The original layer relied on child constructors, then the model's
        # post-init resets. An extra reset here would shift every later RNG draw.
        if architecture_profile == 'current':
            self.reset_parameters()

    def compute_beta(self, hidden_states: torch.Tensor, *, compact: bool = False) -> torch.Tensor:
        """Return sigmoid rates, optionally in compact ``[..., H, D_v]`` form.

        Current models own shared ``[H,D_v]`` rates; historical plain models
        own ``[H,D_v,E]`` rates. Compact output never collapses expert owners.
        """
        beta = self.b_proj(hidden_states).sigmoid()
        if self.architecture_profile == 'historical_8dc05f93':
            return beta.view(*hidden_states.shape[:-1], self.num_heads, self.head_v_dim, self.num_experts)
        beta = beta.view(
            *hidden_states.shape[:-1],
            self.num_heads,
            self.head_v_dim,
        )
        if compact:
            return beta
        return beta[..., None].expand(
            *hidden_states.shape[:-1],
            self.num_heads,
            self.head_v_dim,
            self.num_experts,
        ).contiguous()

    def compute_log_decay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute ``-exp(A_log) * softplus(a_proj(x) + dt_bias)`` in float32.

        The result has shape ``[..., H, D_k]`` and broadcasts over the
        recurrent state's expert and value dimensions.
        """
        alpha_raw = self.a_proj(hidden_states)
        alpha_raw = alpha_raw.view(
            *hidden_states.shape[:-1],
            self.num_heads,
            self.head_k_dim,
        )

        # CUDA uses the existing fused FP32 gate transform automatically. The
        # CPU path remains the direct PyTorch expression, which keeps this
        # layer importable in CPU-only installations where Triton is absent.
        if alpha_raw.is_cuda:
            from fla.ops.swila import fused_swila_log_decay_gate

            return fused_swila_log_decay_gate(
                alpha_raw,
                self.A_log,
                self.dt_bias,
            )
        # FSDP mixed precision can expose BF16 runtime views of these stored
        # FP32 parameters. Follow FLA by promoting the views before exp and
        # softplus so the gate transform and its output remain FP32.
        A = self.A_log.float().exp()
        return -A[:, None] * F.softplus(alpha_raw.float() + self.dt_bias.float())

    def _apply(self, fn, recurse: bool = True):
        """Apply transforms while keeping decay scalars fp32 and no-weight-decay."""
        # Let PyTorch retain its normal parameter/gradient identity handling,
        # but do not round protected values through a lower precision dtype.
        # Inspecting the transformed tensor also handles device moves and
        # leaves meta -> to_empty materialization on PyTorch's normal path.
        protected = set()
        if self.use_forget_gate:
            for parameter in (self.A_log, self.dt_bias):
                protected.add(id(parameter))
                if parameter.grad is not None:
                    protected.add(id(parameter.grad))

        def apply_preserving_decay(tensor):
            transformed = fn(tensor)
            if id(tensor) in protected and transformed.dtype != torch.float32:
                if torch.finfo(transformed.dtype).bits < 32:
                    return tensor.to(device=transformed.device, dtype=torch.float32)
                return transformed.float()
            return transformed

        super()._apply(apply_preserving_decay, recurse=recurse)
        if self.use_forget_gate:
            for parameter in (self.A_log, self.dt_bias):
                if parameter.dtype != torch.float32:
                    parameter.data = parameter.data.float()
                if parameter.grad is not None and parameter.grad.dtype != torch.float32:
                    parameter.grad.data = parameter.grad.data.float()
                parameter._no_weight_decay = True
        if self.temporal:
            self.gate_kv_proj.bias._no_weight_decay = True
            self.gate_q_proj.bias._no_weight_decay = True
            if self.inject_gate_noise:
                self.W_noise_gate_kv.bias._no_weight_decay = True
                self.W_noise_gate_q.bias._no_weight_decay = True
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor,
            torch.Tensor | dict[str, torch.Tensor] | None,
            Cache | None,
            torch.Tensor | None,
        ]
    ):
        """
        Forward pass of Switching Linear Attention.

        Args:
            hidden_states: [B, T, D] - Input tensor
            attention_mask: [B, T] - Optional attention mask
            past_key_values: Optional cache for incremental decoding
            use_cache: Whether to produce a final state when no cache object is
                supplied. Following FLA's cache contract, a supplied
                ``past_key_values`` object is always advanced.
            output_attentions: Whether to return attention weights (responsibilities)

        Returns:
            If return_output_only=True:
                output: [B, T, D] - Output tensor
            Otherwise:
                output: [B, T, D] - Output tensor
                attentions: Optional key responsibilities, or a dictionary
                    containing both r and rho for temporal SwiLA.
                past_key_values: The supplied cache, advanced whenever it is
                    not ``None``.
                aux_loss: Optional router mutual-information loss.

        Diagnostics are optional: use ``capture_swila_diagnostics`` around this
        call (and backward when activation-gradient norms are requested).
        """
        check_recurrent_mode(self.mode)
        capture = active_swila_capture(self)
        if kwargs.get("ignore_attention_mask", False):
            attention_mask = None

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, "attention_mask must have shape [batch, seq_len]"

        cu_seqlens = kwargs.get('cu_seqlens')
        if cu_seqlens is not None:
            if not isinstance(cu_seqlens, torch.Tensor):
                raise TypeError("cu_seqlens must be a torch.Tensor")
            cu_seqlens = cu_seqlens.reshape(-1)
        if attention_mask is not None and any(
            helper is None
            for helper in (unpad_hidden_states, repad_hidden_states, pad_input)
        ):
            raise RuntimeError("attention_mask handling requires FLA padding helpers")

        # Get last state from cache if available
        last_state = None
        if past_key_values is not None:
            if not isinstance(self.layer_idx, int) or self.layer_idx < 0:
                raise ValueError("layer_idx must be a nonnegative integer when caching")
            if get_layer_cache is None or update_layer_cache is None:
                raise RuntimeError("cache handling requires FLA cache helpers")
            last_state = get_layer_cache(self, past_key_values)

        # FLA treats the presence of a cache object as the request to advance
        # it, independently of the model-level ``use_cache`` return policy.
        # Request final recurrent/convolution states in that case so the
        # update contains the state after this call rather than merely
        # incrementing the cache offset while retaining stale tensors.
        output_final_state = bool(use_cache) or past_key_values is not None

        batch_size, seq_len, _ = hidden_states.shape

        # SwiLA retains mask precedence over explicit packed boundaries.
        indices = None
        if unpad_hidden_states is not None:
            hidden_states, indices, cu_seqlens = unpad_hidden_states(
                hidden_states,
                None if attention_mask is not None else cu_seqlens,
                attention_mask,
                seq_len,
            )
        num_sequences = validate_packed_boundaries(hidden_states, cu_seqlens)

        initial_r, initial_rho = None, None
        if last_state is not None:
            if self.temporal:
                recurrent_state, initial_r, initial_rho = last_state['recurrent_state']
            else:
                recurrent_state = last_state['recurrent_state']
        elif self.init_state_mode == 'learnable':
            recurrent_state = self.init_state.expand(num_sequences, -1, -1, -1, -1)
        else:
            recurrent_state = None
        if (initial_r is None) != (initial_rho is None):
            raise ValueError(
                "r_prev_init and rho_prev_init must either both be provided or both be None"
            )

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        pi_conv_state_q, pi_conv_state_k = None, None
        if self.use_short_conv and last_state is not None:
            cached_conv = last_state['conv_state']
            if cached_conv is not None and len(cached_conv) == 5:
                conv_state_q, conv_state_k, conv_state_v, pi_conv_state_q, pi_conv_state_k = cached_conv
            elif cached_conv is not None:
                conv_state_q, conv_state_k, conv_state_v = cached_conv
        validate_sequence_state_counts(
            {
                "initial_state": recurrent_state,
                "r_prev_init": initial_r,
                "rho_prev_init": initial_rho,
                "conv_state_q": conv_state_q,
                "conv_state_k": conv_state_k,
                "conv_state_v": conv_state_v,
                "pi_conv_state_q": pi_conv_state_q,
                "pi_conv_state_k": pi_conv_state_k,
            },
            num_sequences,
        )
        # ShortConvolution.step indexes this public [N, D, W] layout directly.
        # Check every cache before the first convolution can mutate any of them.
        for name, state, channels in (
            ("conv_state_q", conv_state_q, self.key_dim),
            ("conv_state_k", conv_state_k, self.key_dim),
            ("conv_state_v", conv_state_v, self.value_dim),
            ("pi_conv_state_q", pi_conv_state_q, self.pi_input_dim),
            ("pi_conv_state_k", pi_conv_state_k, self.pi_input_dim),
        ):
            if state is None:
                continue
            expected_shape = (num_sequences, channels, self.conv_size)
            if tuple(state.shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(state.shape)}")
            if state.device != hidden_states.device:
                raise ValueError(f"{name} must be on the same device as hidden_states")

        # Apply projections and convolutions
        if self.use_short_conv:
            q_proj_out = self.q_proj(hidden_states)
            k_proj_out = self.k_proj(hidden_states)
            v_proj_out = self.v_proj(hidden_states)
            pi_q_proj_out = self.pi_q_input_proj(hidden_states)
            pi_k_proj_out = self.pi_k_input_proj(hidden_states)

            q, conv_state_q = self.q_conv1d(
                x=q_proj_out,
                cache=conv_state_q,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=k_proj_out,
                cache=conv_state_k,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=v_proj_out,
                cache=conv_state_v,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )

            # Separate conv1d pathway for the expert-prior routers.
            if self.pi_q_conv1d is not None and self.pi_k_conv1d is not None:
                pi_q_conv_out, pi_conv_state_q = self.pi_q_conv1d(
                    x=pi_q_proj_out,
                    cache=pi_conv_state_q,
                    output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens,
                )
                pi_k_conv_out, pi_conv_state_k = self.pi_k_conv1d(
                    x=pi_k_proj_out,
                    cache=pi_conv_state_k,
                    output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens,
                )
            else:
                pi_q_conv_out = pi_q_proj_out
                pi_k_conv_out = pi_k_proj_out
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            pi_q_conv_out = self.pi_q_input_proj(hidden_states)
            pi_k_conv_out = self.pi_k_input_proj(hidden_states)

        # Reshape to heads: [B, T, H, D_head]
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        # Apply QK normalization
        if self.qk_norm == 'l2':
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        beta = self.compute_beta(hidden_states)

        # Compute log decay. Its exponential is in (0, 1] and is shared over D_v.
        if self.use_forget_gate:
            log_decay = self.compute_log_decay(hidden_states)
        else:
            log_decay = None

        pi_q, pi_k, pi_q_input_head, pi_k_input_head, noise_std_k, noise_std_q = compute_router_logits(
            self, pi_q_conv_out, pi_k_conv_out,
        )
        gate_kv, gate_q, gate_noise_std_kv, gate_noise_std_q = compute_temporal_gates(
            self, pi_q_input_head, pi_k_input_head,
        )

        # Auxiliary recurrence outputs used by MI loss and optional diagnostics.
        r_seq, rho_seq = None, None
        u_seq, pi_bar_seq = None, None
        train_mi = self.training and self.mi_loss_weight > 0.0

        effective_mode = self.mode
        if self.mode == 'fused_recurrent' and self.temporal:
            effective_mode = 'fused_recurrent_temporal'

        if effective_mode == 'fused_recurrent_temporal':
            # Fused Triton kernel for SwiLA with temporal transitions
            from fla.ops.swila import fused_recurrent_swila_temporal

            # For fused kernel: squeeze trailing dim from gate [B,T,H,D_v,1] -> [B,T,H,D_v]
            fused_gate_kv = gate_kv.squeeze(-1) if gate_kv is not None else None
            fused_gate_q = gate_q.squeeze(-1) if gate_q is not None else None
            temporal_kwargs = dict(
                q=q,
                k=k,
                v=v,
                beta=beta,
                pi_q=pi_q,
                pi_k=pi_k,
                scale=self.scale,
                initial_state=recurrent_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
                r_prev_init=initial_r,
                rho_prev_init=initial_rho,
                gate_kv=fused_gate_kv,
                gate_q=fused_gate_q,
                cu_seqlens=cu_seqlens,
                log_decay=log_decay,
            )
            o, recurrent_state, r_final, rho_final, r_seq, rho_seq, _, u_seq, pi_bar_seq = fused_recurrent_swila_temporal(
                **temporal_kwargs
            )

            responsibilities = {"r": r_seq, "rho": rho_seq}

            # Update temporal state
            initial_r = r_final
            initial_rho = rho_final

        elif effective_mode == 'fused_recurrent':
            # Fused Triton kernel for SwiLA
            from fla.ops.swila import fused_recurrent_swila

            return_rho_seq_for_swila_mi = (
                (capture is not None and capture.retain_tensors)
                or self.mi_loss_weight > 0.0
            )
            o, recurrent_state, r_seq, rho_seq = fused_recurrent_swila(
                q=q,
                k=k,
                v=v,
                beta=beta,
                pi_q=pi_q,
                pi_k=pi_k,
                scale=self.scale,
                initial_state=recurrent_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_seqlens,
                return_rho_seq=return_rho_seq_for_swila_mi,
                log_decay=log_decay,
            )

            responsibilities = r_seq

        else:
            raise ValueError(f"Unsupported SwiLA layer mode: {self.mode!r}")

        aux_loss = None
        if train_mi or capture is not None:
            collect_router_stats = capture is not None and capture.collect_router_stats
            need_rho = train_mi or collect_router_stats or capture.retain_tensors
            if isinstance(responsibilities, dict):
                r, rho = responsibilities["r"], responsibilities["rho"]
            else:
                r = responsibilities if responsibilities is not None else r_seq
                rho = rho_seq
                if need_rho and r is not None and rho is None:
                    rho = F.softmax(pi_q if train_mi else pi_q.detach(), dim=-1)
            mi_result = compute_swila_mi_loss(
                r, rho, self.num_experts, self.mi_loss_weight, self.training,
                cu_seqlens, return_stats=collect_router_stats,
            )
            if collect_router_stats:
                aux_loss, mi_stats = mi_result
            else:
                aux_loss, mi_stats = mi_result, None
            if capture is not None:
                capture.record(
                    self, pi_q=pi_q, pi_k=pi_k, r=r, rho=rho,
                    responsibilities=responsibilities, gate_kv=gate_kv, gate_q=gate_q,
                    u=u_seq, pi_bar=pi_bar_seq,
                    beta=beta, cu_seqlens=cu_seqlens,
                    aux_loss=aux_loss, mi_stats=mi_stats,
                    noise_std_k=noise_std_k, noise_std_q=noise_std_q,
                    gate_noise_std_kv=gate_noise_std_kv,
                    gate_noise_std_q=gate_noise_std_q,
                )

        # Pack temporal state as tuple for cache (FLA Cache expects tensor or tuple, not dict)
        if self.temporal and isinstance(recurrent_state, torch.Tensor) and initial_r is not None:
            recurrent_state = (recurrent_state, initial_r, initial_rho)

        # Update cache
        if past_key_values is not None:
            update_layer_cache(
                self,
                past_key_values,
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v, pi_conv_state_q,
                            pi_conv_state_k) if self.use_short_conv else None,
                offset=seq_len,
            )

        # Apply output normalization
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        # Reshape and project output
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if capture is not None:
            capture.register_gradients(
                self, q=q, k=k, v=v, beta=beta, pi_q=pi_q, pi_k=pi_k,
                gate_kv=gate_kv, gate_q=gate_q, output=o,
            )

        # Restore the original padded layout for every public or retained
        # token-level result, not only for the layer output.
        if indices is not None:
            o = repad_hidden_states(o, indices, batch_size, seq_len)

        # Return responsibilities if requested
        attentions = responsibilities if output_attentions else None
        if indices is not None:
            if isinstance(attentions, dict):
                attentions = {
                    name: repad_packed_tensor(
                        value,
                        indices,
                        batch_size,
                        seq_len,
                        pad_fn=pad_input,
                    )
                    for name, value in attentions.items()
                }
            else:
                attentions = repad_packed_tensor(
                    attentions,
                    indices,
                    batch_size,
                    seq_len,
                    pad_fn=pad_input,
                )
            if capture is not None:
                capture.repad(
                    self, indices=indices, batch_size=batch_size, seq_len=seq_len,
                    valid_token_mask=attention_mask[:, -seq_len:].bool(),
                    pad_fn=pad_input,
                )

        if self.return_output_only:
            # Store aux_loss with gradient for external collection (e.g. GenericCausalLM,
            # CDAR trainer).  This is safe because return_output_only models do NOT use
            # activation checkpointing — the memory-leak concern only applies to the
            # SwiLAForCausalLM path which returns aux_loss through the 4-tuple instead.
            self._aux_loss = aux_loss
            return o
        return o, attentions, past_key_values, aux_loss

    def reset_parameters(self):
        """Restore SwiLA-specific initialization after generic weight initialization."""
        initialization.initialize_swila_parameters(self)

    def state_size(self, sequence_length: int = None) -> int:
        """Return the size of the recurrent state [B, H, D_v, K, D_k].

        Args:
            sequence_length: Unused; retained for the CDAR model interface.
        """
        return self.num_heads * self.head_v_dim * self.num_experts * self.head_k_dim
