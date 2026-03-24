# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Op registry, input factory, and shape configs for the unified benchmark system.

See ``benchmarks/ops/run.py`` docstring for full usage and how to register new ops.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shape helpers: reusable callables  (B, T, H, D, **kw) -> tuple
# ---------------------------------------------------------------------------


def shape_BTHD(B, T, H, D, **kw):
    return (B, T, H, D)


def shape_BTH(B, T, H, D, **kw):
    return (B, T, H)


def shape_BTD(B, T, H, D, **kw):
    return (B, T, H * D)


def shape_H(B, T, H, D, **kw):
    return (H,)


def shape_HD(B, T, H, D, **kw):
    return (H, D)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

logsigmoid = F.logsigmoid


def sigmoid_transform(t):
    return t.sigmoid()


def logsigmoid_clamp(t):
    return F.logsigmoid(t).clamp_min(-5)


# ---------------------------------------------------------------------------
# TensorSpec: describes how to create one input tensor
# ---------------------------------------------------------------------------


@dataclass
class TensorSpec:
    """Specification for generating a single benchmark input tensor.

    Args:
        shape_fn:       (B, T, H, D, **kw) -> tuple of ints
        requires_grad:  whether the tensor needs gradients
        dtype:          'default' inherits from the benchmark, or 'float32'/'long'
        transform:      applied after randn, e.g. F.logsigmoid
    """
    shape_fn: Callable
    requires_grad: bool = True
    dtype: str = 'default'
    transform: Callable | None = None


# ---------------------------------------------------------------------------
# OpConfig: registry entry for one op
# ---------------------------------------------------------------------------


@dataclass
class OpConfig:
    """Registry entry describing how to benchmark a single op.

    Args:
        name:           display/registry name, e.g. 'chunk_gla'
        import_path:    Python module path, e.g. 'fla.ops.gla'
        inputs:         param_name -> TensorSpec mapping
        func_name:      actual function attribute name if different from *name*
        extra_kwargs:   constant keyword args passed to the op
        output_is_tuple: True if output[0] is the tensor to .backward()
        skip_backward:  True to skip fwdbwd mode
        post_init:      callable(inputs_dict, B, T, H, D, **kw) for custom mutation
        category:       grouping label
        dim_constraints: e.g. {'D': [64, 128]} — skip shapes that don't match
    """
    name: str
    import_path: str
    inputs: dict[str, TensorSpec]
    func_name: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    output_is_tuple: bool = True
    skip_backward: bool = False
    post_init: Callable | None = None
    category: str = ''
    dim_constraints: dict | None = None


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, OpConfig] = {}


def register_op(config: OpConfig) -> None:
    _REGISTRY[config.name] = config


def get_op(name: str) -> OpConfig:
    if name not in _REGISTRY:
        raise KeyError(f"Op '{name}' not registered. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_ops() -> list[str]:
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Shape configs
# ---------------------------------------------------------------------------

SHAPE_CONFIGS = {
    'B1_T8192_H96_D128':  {'B': 1,  'T': 8192,  'H': 96, 'D': 128},
    'B2_T16384_H16_D128': {'B': 2,  'T': 16384, 'H': 16, 'D': 128},
    'B4_T2048_H16_D128':  {'B': 4,  'T': 2048,  'H': 16, 'D': 128},
    'B4_T4096_H64_D128':  {'B': 4,  'T': 4096,  'H': 64, 'D': 128},
    'B8_T2048_H32_D256':  {'B': 8,  'T': 2048,  'H': 32, 'D': 256},
    'B8_T1024_H8_D64':    {'B': 8,  'T': 1024,  'H': 8,  'D': 64},
}


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------


def generate_inputs(
    config: OpConfig,
    B: int, T: int, H: int, D: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = 'cuda',
    **extra_shape_kw,
) -> dict[str, torch.Tensor]:
    """Create input tensors for *config* at the given shape.

    Returns a dict mapping parameter names to tensors.
    Raises ValueError if dim_constraints are not satisfied (caller should skip).
    """
    # Check dim constraints
    if config.dim_constraints:
        shape_vals = {'B': B, 'T': T, 'H': H, 'D': D, **extra_shape_kw}
        for dim_name, allowed in config.dim_constraints.items():
            val = shape_vals.get(dim_name)
            if val is not None and val not in allowed:
                raise ValueError(
                    f"Op '{config.name}' requires {dim_name} in {allowed}, got {val}"
                )

    inputs = {}
    for param_name, spec in config.inputs.items():
        shape = spec.shape_fn(B, T, H, D, **extra_shape_kw)

        # Determine dtype
        if spec.dtype == 'default':
            tensor_dtype = dtype
        elif spec.dtype == 'float32':
            tensor_dtype = torch.float32
        elif spec.dtype == 'long':
            tensor_dtype = torch.long
        else:
            tensor_dtype = dtype

        if tensor_dtype == torch.long:
            tensor = torch.randint(0, 10, shape, dtype=tensor_dtype, device=device)
        else:
            tensor = torch.randn(shape, dtype=tensor_dtype, device=device)

        if spec.transform is not None:
            tensor = spec.transform(tensor)

        if spec.requires_grad and tensor.is_floating_point():
            tensor = tensor.requires_grad_(True)

        inputs[param_name] = tensor

    # Custom post-init mutation
    if config.post_init is not None:
        config.post_init(inputs, B=B, T=T, H=H, D=D, **extra_shape_kw)

    return inputs


# ===========================================================================
# Op registrations
# ===========================================================================

# --- A: Simple qkv (no extra inputs) ---

_simple_qkv = {
    'q': TensorSpec(shape_BTHD),
    'k': TensorSpec(shape_BTHD),
    'v': TensorSpec(shape_BTHD),
}

register_op(OpConfig(
    name='chunk_retention',
    import_path='fla.ops.retention',
    inputs={**_simple_qkv},
    category='simple_qkv',
))

register_op(OpConfig(
    name='chunk_linear_attn',
    import_path='fla.ops.linear_attn',
    inputs={**_simple_qkv},
    category='simple_qkv',
))

# --- B: +elem gate (g=[B,T,H,D] with logsigmoid_clamp) ---

register_op(OpConfig(
    name='chunk_gla',
    import_path='fla.ops.gla',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTHD, transform=logsigmoid_clamp),
    },
    category='elem_gate',
))

# --- C: +beta (beta=[B,T,H] with sigmoid) ---

register_op(OpConfig(
    name='chunk_delta_rule',
    import_path='fla.ops.delta_rule',
    inputs={
        **_simple_qkv,
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    category='beta',
))

# --- D: +gate + beta ---

register_op(OpConfig(
    name='chunk_gdn',
    import_path='fla.ops.gated_delta_rule',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTH, transform=logsigmoid),
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    func_name='chunk_gated_delta_rule',
    extra_kwargs={'use_qk_l2norm_in_kernel': True},
    category='gate_beta',
))

register_op(OpConfig(
    name='chunk_kda',
    import_path='fla.ops.kda',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTHD, transform=logsigmoid),
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True, 'safe_gate': True, 'lower_bound': -5},
    category='gate_beta',
))

# --- E: +head gate (g=[B,T,H] with logsigmoid) ---

register_op(OpConfig(
    name='chunk_simple_gla',
    import_path='fla.ops.simple_gla',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTH, transform=logsigmoid),
    },
    category='head_gate',
))

# --- F: RWKV ---


def _rwkv7_post_init(inputs, B, T, H, D, **kw):
    """RWKV7 needs a/b to be initialized as small positive values."""
    with torch.no_grad():
        inputs['a'] = (torch.randn_like(inputs['a']) * 0.1).requires_grad_(True)
        inputs['b'] = (torch.randn_like(inputs['b']) * 0.1).requires_grad_(True)


register_op(OpConfig(
    name='chunk_rwkv6',
    import_path='fla.ops.rwkv6',
    inputs={
        'r': TensorSpec(shape_BTHD),
        'k': TensorSpec(shape_BTHD),
        'v': TensorSpec(shape_BTHD),
        'w': TensorSpec(shape_BTHD, transform=logsigmoid),
        'u': TensorSpec(shape_HD, requires_grad=False),
    },
    category='rwkv',
))

register_op(OpConfig(
    name='chunk_rwkv7',
    import_path='fla.ops.rwkv7',
    inputs={
        'r': TensorSpec(shape_BTHD),
        'w': TensorSpec(shape_BTHD, transform=logsigmoid),
        'k': TensorSpec(shape_BTHD),
        'v': TensorSpec(shape_BTHD),
        'a': TensorSpec(shape_BTHD),
        'b': TensorSpec(shape_BTHD),
    },
    post_init=_rwkv7_post_init,
    category='rwkv',
))

# --- H: Comba ---

register_op(OpConfig(
    name='chunk_comba',
    import_path='fla.ops.comba',
    inputs={
        **_simple_qkv,
        'p': TensorSpec(shape_BTHD),
        'g': TensorSpec(shape_BTH, transform=logsigmoid),
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True},
    category='comba',
))

# --- I: HGRN (x, g only, no qkv) ---

register_op(OpConfig(
    name='fused_recurrent_hgrn',
    import_path='fla.ops.hgrn',
    inputs={
        'x': TensorSpec(shape_BTD),
        'g': TensorSpec(shape_BTD, transform=logsigmoid),
    },
    category='hgrn',
))

# --- J: Generalized delta rule (DPLR) ---

register_op(OpConfig(
    name='chunk_dplr_delta_rule',
    import_path='fla.ops.generalized_delta_rule',
    inputs={
        **_simple_qkv,
        'a': TensorSpec(shape_BTHD),
        'b': TensorSpec(shape_BTHD),
        'gk': TensorSpec(shape_BTHD, transform=logsigmoid),
    },
    category='gen_delta',
))

# --- K: Lightning attention (needs layer_idx, num_layers) ---

register_op(OpConfig(
    name='chunk_lightning_attn',
    import_path='fla.ops.lightning_attn',
    inputs={**_simple_qkv},
    extra_kwargs={'layer_idx': 0, 'num_layers': 12},
    category='lightning',
))

# --- L: Attention baselines ---

register_op(OpConfig(
    name='parallel_attn',
    import_path='fla.ops.attn',
    inputs={**_simple_qkv},
    output_is_tuple=False,
    category='attn',
))


register_op(OpConfig(
    name='flash_attn',
    import_path='flash_attn',
    inputs={**_simple_qkv},
    func_name='flash_attn_func',
    extra_kwargs={'causal': True},
    output_is_tuple=False,
    category='flash_attn',
))
