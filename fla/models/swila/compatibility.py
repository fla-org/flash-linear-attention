# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Input-only metadata accepted from historical SwiLA model configurations."""

from fla.modules.swila.argument_checks import check_recurrent_mode

# Serialized checkpoint identities are input compatibility, not import aliases.
SWILA_MODEL_TYPES = frozenset({"swila", "sla"})
LEGACY_ARCHITECTURE_NAMES = {
    "SLAConfig": "SwiLAConfig",
    "SLABlock": "SwiLABlock",
    "SLAModel": "SwiLAModel",
    "SLAForCausalLM": "SwiLAForCausalLM",
    "SLAPreTrainedModel": "SwiLAPreTrainedModel",
}

LEGACY_DEER_SWILA_FIELDS = (
    "num_deer_iterations", "chunk_size", "chunk_backend", "pred_init_mode",
    "full_jacobian",
)

LEGACY_FIXED_SWILA_VALUES = {
    "b_granularity": "value",
    "alpha_granularity": "key",
    "decay_residual_mode": "post_decay",
    "skip_reset_parameters": False,
    "pi_input_proj_narrow": True,
    "pi_init_orthogonal": False,
    "conv_activation": "silu",
    "pi_conv_activation": "silu",
    "qk_activation": "silu",
    "init_state_learnable_init": "zeros",
    "pi_activation": "softmax",
    "use_input_pi": True,
    "gate_init_mode": "bias",
    "factorial": True,
    "clip_threshold": None,
    "tie_qk_proj": False,
    "tie_responsibilities": False,
    "temporal_coupling": "reset_gate",
    "temporal_sticky_init": 1.0,
    "fix_transition_identity": True,
}
LEGACY_INERT_SWILA_FIELDS = (
    "gate_min_half_life", "gate_max_half_life", "l_max",
    "responsibilities_use_post_decay",
)
LEGACY_BEHAVIOR_SWILA_FIELDS = (
    "factorial", "clip_threshold", "tie_qk_proj", "tie_responsibilities",
    "temporal_coupling", "temporal_sticky_init", "fix_transition_identity",
)


def validate_legacy_fixed_swila_field(key: str, value) -> None:
    """Accept equivalent legacy choices without restoring configuration fields."""
    expected = LEGACY_FIXED_SWILA_VALUES[key]
    # Preserve the former constructor's equality rule for sticky/transition
    # metadata; booleans and None still require their exact historical values.
    if key in LEGACY_BEHAVIOR_SWILA_FIELDS:
        matches = value is expected if expected is None or isinstance(expected, bool) else value == expected
        if not matches:
            raise ValueError(f"Legacy SwiLAConfig fields request unsupported behavior: {key} must be {expected!r}")
    elif type(value) is not type(expected) or value != expected:
        raise ValueError(f"{key} has been removed; only legacy {expected!r} metadata is accepted, got {value!r}.")


def discard_legacy_swila_fields(fields: dict, *, mode="fused_recurrent") -> None:
    """Discard inert metadata only after validating the recurrent algorithm."""
    check_recurrent_mode(mode)
    for key in LEGACY_FIXED_SWILA_VALUES:
        if key in fields:
            validate_legacy_fixed_swila_field(key, fields.pop(key))
    for key in LEGACY_INERT_SWILA_FIELDS:
        fields.pop(key, None)
    for key in LEGACY_DEER_SWILA_FIELDS:
        fields.pop(key, None)
