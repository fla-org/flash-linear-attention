# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).


"""Constructor argument checks for SwitchingLinearAttention."""

import math
from functools import wraps
from inspect import signature
from types import SimpleNamespace


def check_architecture_profile(*, architecture_profile, temporal, use_forget_gate,
                               inject_noise, inject_gate_noise,
                               pi_align_qk_init, b_low_rank_proj_type, pi_low_rank_proj_type,
                               low_rank_proj_type, b_proj_rank, pi_proj_rank) -> None:
    """Opt into the original plain-SwiLA architecture without changing defaults.

    The historical profile restores expert-specific SwiGLU beta and the old
    reset policy. Dimensions remain configurable for small correctness fixtures.
    Fixed numerical improvements (raw logits and FP32 MI) are not reverted.
    """
    if architecture_profile not in ('current', 'historical_8dc05f93'):
        raise ValueError("architecture_profile must be 'current' or 'historical_8dc05f93'")
    if architecture_profile == 'current':
        return
    if temporal or use_forget_gate or inject_noise or inject_gate_noise:
        raise ValueError("historical_8dc05f93 requires plain SwiLA without noise")
    pi_type = low_rank_proj_type if pi_low_rank_proj_type in (None, 'null') else pi_low_rank_proj_type
    pi_type = 'swiglu' if pi_type in (None, 'null') else pi_type
    if b_low_rank_proj_type != 'swiglu' or pi_type != 'swiglu' or b_proj_rank <= 0 or pi_proj_rank <= 0:
        raise ValueError("historical_8dc05f93 requires positive-rank SwiGLU beta and router projections")
    if pi_align_qk_init:
        raise ValueError("historical_8dc05f93 requires pi_align_qk_init=False")


def check_recurrent_mode(mode) -> None:
    """Reject retired execution modes rather than silently changing algorithms."""
    if mode != 'fused_recurrent':
        raise ValueError(f"mode must be 'fused_recurrent'; chunk/DEER support has been removed, got {mode!r}.")


def check_swila_args(args: SimpleNamespace) -> None:
    """Validate bound constructor arguments without modifying them or the layer."""
    check_recurrent_mode(args.mode)
    check_architecture_profile(**{key: getattr(args, key) for key in (
        'architecture_profile', 'temporal', 'use_forget_gate', 'inject_noise',
        'inject_gate_noise', 'pi_align_qk_init',
        'b_low_rank_proj_type', 'pi_low_rank_proj_type', 'low_rank_proj_type',
        'b_proj_rank', 'pi_proj_rank',
    )})
    if args.qk_norm not in ('l2', None):
        raise ValueError(f"qk_norm must be 'l2' or None, got {args.qk_norm!r}.")
    if args.d_model is None or args.d_model <= 0:
        raise ValueError(f"d_model must be a positive integer, got {args.d_model!r}")
    if args.num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {args.num_heads!r}")
    if args.num_experts < 1:
        raise ValueError(f"num_experts must be at least 1, got {args.num_experts!r}")
    assert args.num_experts <= 8, f"fused_recurrent SwiLA requires num_experts <= 8, got {args.num_experts!r}"
    if args.expand_k <= 0 or args.expand_v <= 0:
        raise ValueError("expand_k and expand_v must be positive")
    for name in ('pi_proj_rank', 'b_proj_rank', 'alpha_proj_rank'):
        rank = getattr(args, name)
        if rank < 0:
            raise ValueError(f"{name} must be nonnegative, got {rank!r}")
    if not math.isfinite(args.gate_noise_init_std) or args.gate_noise_init_std <= 0:
        raise ValueError(f"gate_noise_init_std must be positive, got {args.gate_noise_init_std!r}")

    base_proj_type = 'swiglu' if args.low_rank_proj_type in (None, 'null') else args.low_rank_proj_type
    for name, fallback in (
        ('low_rank_proj_type', 'swiglu'),
        ('pi_low_rank_proj_type', base_proj_type),
        ('b_low_rank_proj_type', 'linear'),
        ('alpha_low_rank_proj_type', 'linear'),
    ):
        value = getattr(args, name)
        resolved = fallback if value in (None, 'null') else value
        if resolved not in ('linear', 'swiglu'):
            raise ValueError(f"{name} must be 'linear' or 'swiglu', got {resolved!r}")

    assert args.init_state_mode in ['zeros', 'learnable'], (
        f"init_state_mode must be 'zeros' or 'learnable', got '{args.init_state_mode}'"
    )
    key_dim = int(args.d_model * args.expand_k)
    value_dim = int(args.d_model * args.expand_v)
    if key_dim < 1 or value_dim < 1:
        raise ValueError("expanded key and value dimensions must be positive")
    assert key_dim % args.num_heads == 0, (
        f"key dim must be divisible by num_heads of {args.num_heads}"
    )
    assert value_dim % args.num_heads == 0, (
        f"value dim must be divisible by num_heads of {args.num_heads}"
    )
    if (
        args.pi_proj_rank > 0
        and args.pi_proj_rank % args.num_heads != 0
    ):
        raise ValueError("pi_proj_rank must be divisible by num_heads for narrow projections")


def validate_swila_args(init):
    """Check constructor inputs once, preserving its signature and call arguments."""
    init_signature = signature(init)

    @wraps(init)
    def checked_init(self, *args, **kwargs):
        bound = init_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        check_swila_args(SimpleNamespace(**bound.arguments))
        return init(self, *args, **kwargs)

    return checked_init
