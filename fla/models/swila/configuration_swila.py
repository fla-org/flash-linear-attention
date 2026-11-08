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
Switching Linear Attention Model Configuration
"""

import math
import warnings
from typing import Literal

from transformers.configuration_utils import PreTrainedConfig

from fla.models.swila.compatibility import (
    LEGACY_ARCHITECTURE_NAMES,
    LEGACY_DEER_SWILA_FIELDS,
    LEGACY_FIXED_SWILA_VALUES,
    LEGACY_INERT_SWILA_FIELDS,
    discard_legacy_swila_fields,
    validate_legacy_fixed_swila_field,
)
from fla.models.swila.hybrid_config import HybridAttentionConfig, normalize_hybrid_attention_config
from fla.modules.swila.argument_checks import check_architecture_profile, check_recurrent_mode


class SwiLAConfig(PreTrainedConfig):
    """
    Configuration class for Switching Linear Attention (SwiLA) model.

    Args:
        # SwiLA attention parameters
        mode (str): Attention computation mode: 'fused_recurrent'.
            Default: 'fused_recurrent'. The PyTorch reference is not a layer mode.
        hidden_size (int): Dimension of the hidden representations. Default: 2048
        expand_k (float): Expansion ratio for key dimension. Default: 1.0
        expand_v (float): Expansion ratio for value dimension. Default: 1.0
        num_heads (int): Number of attention heads. Default: 16
        num_experts (int): Number of experts in the mixture. Default: 4
        architecture_profile (str): ``'current'`` keeps shared value beta and
            calibrated initialization. Explicit ``'historical_8dc05f93'`` uses
            expert-specific SwiGLU beta and the original independent Kaiming
            router/beta reset order. Its projection and initialization options
            must agree with the profile; it is not inferred from old checkpoints.
        use_gate (bool): Whether to use output gating. Default: False
        use_short_conv (bool): Whether to use short convolutions. Default: False
        conv_size (int): Kernel size for convolutions. Default: 4
        conv_bias (bool): Whether to use bias in convolutions. Default: False
        qk_norm (str | None): Normalization for Q/K ('l2' or None).
            Signed-sum normalization ('sum') is no longer supported.
            Default: 'l2'
        norm_eps (float): Epsilon for normalization. Default: 1e-5
        scale (float | None): Attention scale. Default: None (1/sqrt(D_k))
        return_output_only (bool): Reserved layer-return metadata. The public
            ``SwiLAForCausalLM`` stack consumes the layer's full output contract,
            so this must be ``False``. Default: False
        use_pi_conv (bool): Apply short convolution to pi-path features before
            the terminal logit projection. Default: False
        pi_proj_rank (int): Low-rank width for the global pi projection. Zero
            selects a shared per-head Linear instead. Default: 128
        b_proj_rank (int): Low-rank width for beta. Zero selects a direct
            Linear. Default: 128
        low_rank_proj_type (str): Base projection type used by pi when its
            projection-specific override is not set. 'swiglu' uses
            out(silu(gate(x)) * feat(x)); 'linear' uses up(down(x)).
            Default: 'swiglu'.
        pi_low_rank_proj_type (str | None): Projection override for pi_q/pi_k.
            Defaults to low_rank_proj_type.
        b_low_rank_proj_type (str | None): Projection override for beta.
            Defaults and falls back to 'linear'.
        alpha_low_rank_proj_type (str | None): Projection override for alpha.
            Defaults and falls back to 'linear'.


        init_state_mode (str): ``'zeros'`` uses an implicit zero state;
            ``'learnable'`` adds a trainable state initialized to zeros.
            Default: ``'zeros'``.
        temporal (bool): Enable temporal (rSLDS) filtering with exact identity
            transitions. Default: False

        inject_noise (bool): Add Gaussian noise to pi during training. The
            learned standard deviation starts at pi_noise_init_std. Default: False
        inject_gate_noise (bool): Add noise to gate logits for load balancing (reset_gate mode). Default: False

        pi_init_logit_std (float): Approximate target standard deviation of raw
            pi logits for scaled normal initialization without pi convolution.
            Default: 0.1
        pi_align_qk_init (bool): Initialize the query router once and copy its
            input projection, optional convolution, and terminal projection to
            the key router. Parameters remain separate. Default: True
        pi_noise_init_std (float): Initial standard deviation of injected pi
            router noise when ``inject_noise=True``. Default: 0.02
        gate_init_bias (float): Initial temporal reset-gate logit bias.
            sigmoid(bias) weights fresh pi and 1-sigmoid(bias) weights the
            previous responsibility. Every value-channel bias starts at this
            scalar value.
            Default: -3.0
        gate_noise_init_std (float): Initial standard deviation of injected
            temporal-gate logit noise when ``inject_gate_noise=True``.
            Default: 0.02
        mi_loss_weight (float): MI-based auxiliary loss weight. Default: 0.0
        use_forget_gate (bool): Enable Mamba-style input-dependent state decay.
            Default: False
        alpha_proj_rank (int): Low-rank width for alpha. Zero selects a direct
            Linear. Default: 16
        # Model architecture parameters
        hidden_ratio (int | None): Ratio for computing intermediate_size. Default: 4
        intermediate_size (int | None): MLP intermediate dimension. Default: None
        hidden_act (str): Activation function for MLP. Default: 'swish'
        num_hidden_layers (int): Number of decoder layers. Default: 24
        attn (dict | list[dict] | None): Optional FLA hybrid attention config.
            Layer indices in each specification use Transformer attention instead
            of SwiLA. Default: None.
        max_position_embeddings (int | None): Maximum position count used by
            optional hybrid Transformer attention. Default: None.
        use_cache (bool): Whether to use KV cache. Default: True
        vocab_size (int): Vocabulary size. Default: 4096

        # Optimization flags
        fuse_norm (bool): Whether to fuse normalization ops. Default: True
        fuse_swiglu (bool): Whether to fuse SwiGLU ops. Default: True
        fuse_cross_entropy (bool): Whether to fuse cross entropy. Default: True
        fuse_linear_cross_entropy (bool): Whether to fuse linear cross entropy. Default: False
        use_l2warp (bool): Whether to use L2 warp. Default: False

        # Token IDs
        pad_token_id (int | None): Padding token ID. Default: None
        bos_token_id (int): Beginning of sequence token ID. Default: 1
        eos_token_id (int): End of sequence token ID. Default: 2
        tie_word_embeddings (bool): Whether to tie input/output embeddings. Default: False
        initializer_range (float): Standard deviation used by generic model
            initialization. SwiLA reset_parameters subsequently overrides the
            pi, beta, alpha, temporal-gate, and router-noise pathways with their
            specialized initialization schemes. Default: 0.02
        downstream_config (dict, Optional):
            Opaque downstream metadata, preserved only when supplied. It does not configure SwiLA behavior.

    Initialization behavior (``architecture_profile='current'``):
        Pi input/down weights preserve unit feature scale, while centered and
        variance-corrected terminal rows produce nearly uniform but nonidentical
        expert priors. Beta uses terminal scale 0.1 so sigmoid(beta) starts near
        0.5 and is shared over experts. Alpha uses
        terminal scale 0.05, per-head A_log=0, and inverse-softplus time-step
        biases sampled log-uniformly from [1e-3, 1e-1], shared over experts.
        A_log and dt_bias remain float32 and are
        excluded from weight decay. Router-noise weights start at zero and their
        bias makes the initial standard deviation equal pi_noise_init_std.
        Temporal reset-gate weights start at zero and every bias starts at
        gate_init_bias; optional temporal-gate noise starts at
        gate_noise_init_std. Temporal-gate and gate-noise biases are excluded
        from weight decay because they encode initialized time/noise scales.

        Beta is per value channel, decay is per key channel, and residuals
        use the decayed state. Historical canonical selector metadata is
        accepted and discarded; conflicting selector values are rejected.

        Legacy config objects missing the pi initialization fields are resolved by
        the model as 0.1/True/0.02. Pin them explicitly in experiment configs
        when the raw config text, rather than the resolved config, defines run
        identity or reproducibility.

        Historical fixed-choice fields are input-only metadata: equivalent
        values are accepted and discarded, and unsupported choices fail closed.
        Inert fields, including l_max, are discarded. The accepted metadata is
        defined in fla.models.swila.compatibility; trained state-dict loading is
        unchanged.

        Retired DEER metadata is accepted and discarded only for recurrent
        configurations. A chunk configuration is rejected, including when a
        loading override requests recurrent mode; algorithms are never silently
        substituted.

        The historical plain profile instead uses per-value, per-expert beta
        and independent, uncentered Kaiming router weights. It preserves the
        original model reset/RNG lifecycle, except learned memory is also
        zeroed after meta materialization. Loaded weights remain protected.
        Raw router logits and FP32 MI arithmetic are retained numerical fixes.
        Any legacy b_granularity metadata conflicts with this explicit profile
        and is rejected rather than silently converting checkpoint weights.
    """

    model_type = "swila"
    keys_to_ignore_at_inference = ["past_key_values"]
    _ARCHITECTURE_FIELDS = (
        "architecture_profile", "temporal", "use_forget_gate", "inject_noise",
        "inject_gate_noise", "pi_align_qk_init",
        "b_low_rank_proj_type", "pi_low_rank_proj_type", "low_rank_proj_type",
        "b_proj_rank", "pi_proj_rank",
    )

    def __setattr__(self, key, value):
        # Hugging Face applies loading overrides and updates through setattr
        # after construction. Consume equivalent retired metadata without
        # restoring removed configuration attributes or serialized keys.
        if key == "model_type" and value == "sla":
            value = "swila"
        if key == "architectures" and value is not None:
            value = [LEGACY_ARCHITECTURE_NAMES.get(name, name) for name in value]
        if key in LEGACY_FIXED_SWILA_VALUES:
            if key == "b_granularity" and getattr(self, "architecture_profile", "current") == "historical_8dc05f93":
                raise ValueError("b_granularity metadata conflicts with historical expert-specific beta; use architecture_profile")
            validate_legacy_fixed_swila_field(key, value)
            return
        if key in LEGACY_INERT_SWILA_FIELDS:
            return
        if key in LEGACY_DEER_SWILA_FIELDS:
            check_recurrent_mode(getattr(self, "mode", "fused_recurrent"))
            return
        if key == "temporal_transition_mode":
            raise TypeError("Unsupported SwiLAConfig keys: ['temporal_transition_mode']")
        if key == "mode":
            check_recurrent_mode(value)
        if key == "qk_norm" and value not in ("l2", None):
            raise ValueError(f"qk_norm must be 'l2' or None, got {value!r}.")
        if key == "architecture_profile" and value not in ("current", "historical_8dc05f93"):
            raise ValueError("architecture_profile must be 'current' or 'historical_8dc05f93'")
        if key in self._ARCHITECTURE_FIELDS and "architecture_profile" in self.__dict__:
            options = {name: getattr(self, name) for name in self._ARCHITECTURE_FIELDS}
            options[key] = value
            check_architecture_profile(**options)
        if key == "attn":
            value = normalize_hybrid_attention_config(value, num_hidden_layers=self.num_hidden_layers)
        super().__setattr__(key, value)

    def update(self, config_dict):
        """Prevalidate recurrent mode, profile and legacy metadata atomically.

        Other fields retain ordinary HF update semantics; this is not a
        transaction/rollback guarantee for unrelated invalid attributes.
        """
        check_recurrent_mode(self.mode)
        check_recurrent_mode(config_dict.get("mode", self.mode))
        architecture = {name: config_dict.get(name, getattr(self, name)) for name in self._ARCHITECTURE_FIELDS}
        check_architecture_profile(**architecture)
        # Validate legacy metadata against the final profile before applying any
        # update; dictionary order must not admit conflicting beta ownership.
        if architecture["architecture_profile"] == "historical_8dc05f93" and "b_granularity" in config_dict:
            raise ValueError("b_granularity metadata conflicts with historical expert-specific beta; use architecture_profile")
        for key, value in config_dict.items():
            if key in LEGACY_FIXED_SWILA_VALUES:
                validate_legacy_fixed_swila_field(key, value)
        for key, value in config_dict.items():
            if key in LEGACY_FIXED_SWILA_VALUES or key in LEGACY_INERT_SWILA_FIELDS or key in LEGACY_DEER_SWILA_FIELDS:
                # Already checked against the final profile. Discard directly
                # instead of consulting the old profile through __setattr__.
                continue
            if key in self._ARCHITECTURE_FIELDS:
                super().__setattr__(key, value)
            else:
                setattr(self, key, value)

    def update_from_string(self, update_str):
        """Reject retired modes before HF applies any string-update fields."""
        updates = dict(item.split("=") for item in update_str.split(","))
        check_recurrent_mode(self.mode)
        check_recurrent_mode(updates.get("mode", self.mode))
        updates = {key: value for key, value in updates.items() if key not in LEGACY_DEER_SWILA_FIELDS}
        if updates:
            return super().update_from_string(",".join(f"{key}={value}" for key, value in updates.items()))

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # Validate both identities before merging overrides. A historical
        # chunk checkpoint cannot be reinterpreted as recurrent during loading.
        check_recurrent_mode(config_dict.get("mode", "fused_recurrent"))
        check_recurrent_mode(kwargs.get("mode", "fused_recurrent"))
        config_dict = dict(config_dict)
        # HF otherwise returns unknown override keys as unused without validation.
        if "temporal_transition_mode" in kwargs:
            raise TypeError("Unsupported SwiLAConfig keys: ['temporal_transition_mode']")
        if (kwargs.get("architecture_profile", config_dict.get("architecture_profile")) == "historical_8dc05f93"
                and ("b_granularity" in kwargs or "b_granularity" in config_dict)):
            raise ValueError("b_granularity metadata conflicts with historical expert-specific beta; use architecture_profile")
        discard_legacy_swila_fields(kwargs)
        if "downstream_config" in kwargs:
            config_dict["downstream_config"] = kwargs.pop("downstream_config")
        if (config_dict.get("architecture_profile") == "historical_8dc05f93"
                or kwargs.get("architecture_profile") == "historical_8dc05f93"):
            # HF applies overrides one attribute at a time; constructing the
            # architecture tuple together avoids transient invalid combinations.
            for name in cls._ARCHITECTURE_FIELDS:
                if name in kwargs:
                    config_dict[name] = kwargs.pop(name)
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        # SwiLA attention parameters
        mode: Literal["fused_recurrent"] = "fused_recurrent",
        hidden_size: int = 2048,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 16,
        num_experts: int = 4,
        use_gate: bool = False,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        qk_norm: Literal["l2"] | None = "l2",
        norm_eps: float = 1e-5,
        scale: float | None = None,
        return_output_only: bool = False,
        use_pi_conv: bool = False,
        pi_proj_rank: int = 128,
        b_proj_rank: int = 128,
        low_rank_proj_type: str = "swiglu",
        pi_low_rank_proj_type: str | None = None,
        b_low_rank_proj_type: str | None = "linear",
        alpha_low_rank_proj_type: str | None = "linear",

        init_state_mode: str = "zeros",
        temporal: bool = False,

        inject_noise: bool = False,
        inject_gate_noise: bool = False,

        pi_init_logit_std: float = 0.1,
        pi_align_qk_init: bool = True,
        pi_noise_init_std: float = 0.02,
        gate_init_bias: float = -3.0,
        gate_noise_init_std: float = 0.02,
        mi_loss_weight: float = 0.0,
        use_forget_gate: bool = False,
        alpha_proj_rank: int = 16,
        # Model architecture parameters
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 24,
        attn: HybridAttentionConfig = None,
        max_position_embeddings: int | None = None,
        use_cache: bool = True,
        vocab_size: int = 4096,
        # Optimization flags
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        # Token IDs
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        *,
        architecture_profile: Literal["current", "historical_8dc05f93"] = "current",
        **kwargs,
    ):
        if architecture_profile == "historical_8dc05f93" and "b_granularity" in kwargs:
            raise ValueError("b_granularity metadata conflicts with historical expert-specific beta; use architecture_profile")
        discard_legacy_swila_fields(kwargs, mode=mode)
        check_architecture_profile(
            architecture_profile=architecture_profile, temporal=temporal,
            use_forget_gate=use_forget_gate, inject_noise=inject_noise,
            inject_gate_noise=inject_gate_noise,
            pi_align_qk_init=pi_align_qk_init, b_low_rank_proj_type=b_low_rank_proj_type,
            pi_low_rank_proj_type=pi_low_rank_proj_type, low_rank_proj_type=low_rank_proj_type,
            b_proj_rank=b_proj_rank, pi_proj_rank=pi_proj_rank,
        )
        # A full Transformers export includes base-config fields in addition
        # to SwiLA's constructor arguments. Accept that version's serialized
        # base fields without accepting arbitrary unknown SwiLA options.
        allowed_metadata_keys = set(PreTrainedConfig().to_dict()) | {
            "_attn_implementation_autoset",
            "_commit_hash",
            "_name_or_path",
            "architectures",
            "attn_implementation",
            "auto_map",
            "downstream_config",
            "dtype",
            "experts_implementation",
            "model_type",
            "name_or_path",
            "num_labels",
            "torch_dtype",
            "transformers_version",
        }
        unsupported_keys = sorted(set(kwargs) - allowed_metadata_keys)
        if unsupported_keys:
            raise TypeError(f"Unsupported SwiLAConfig keys: {unsupported_keys}")
        if return_output_only:
            raise ValueError("return_output_only=True is unsupported by SwiLAConfig")
        if (
            not math.isfinite(gate_noise_init_std)
            or gate_noise_init_std <= 0
        ):
            raise ValueError("gate_noise_init_std must be positive")

        # SwiLA attention parameters
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.qk_norm = qk_norm
        self.norm_eps = norm_eps
        self.scale = scale
        self.return_output_only = return_output_only
        self.use_pi_conv = use_pi_conv
        self.pi_proj_rank = pi_proj_rank
        self.b_proj_rank = b_proj_rank
        self.low_rank_proj_type = low_rank_proj_type
        self.pi_low_rank_proj_type = (
            pi_low_rank_proj_type if pi_low_rank_proj_type is not None else low_rank_proj_type
        )
        self.b_low_rank_proj_type = (
            b_low_rank_proj_type if b_low_rank_proj_type is not None else "linear"
        )
        self.alpha_low_rank_proj_type = (
            alpha_low_rank_proj_type if alpha_low_rank_proj_type is not None else "linear"
        )

        self.init_state_mode = init_state_mode
        self.temporal = temporal

        self.inject_noise = inject_noise
        self.inject_gate_noise = inject_gate_noise

        self.pi_init_logit_std = pi_init_logit_std
        self.pi_align_qk_init = pi_align_qk_init
        self.pi_noise_init_std = pi_noise_init_std
        self.gate_init_bias = gate_init_bias
        self.gate_noise_init_std = gate_noise_init_std
        self.mi_loss_weight = mi_loss_weight
        self.use_forget_gate = use_forget_gate
        self.alpha_proj_rank = alpha_proj_rank
        # Register after dependent fields so public mutations can validate
        # the complete architecture profile before changing it.
        self.architecture_profile = architecture_profile
        # Model architecture parameters
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.attn = attn
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.vocab_size = vocab_size

        # Optimization flags
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp

        # Initialization
        self.initializer_range = initializer_range

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError("fuse_cross_entropy and fuse_linear_cross_entropy cannot both be True")
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
