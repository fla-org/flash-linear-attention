# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

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


def shape_D(B, T, H, D, **kw):
    return (D,)


def shape_LBTD(B, T, H, D, L=None, **kw):
    """AttnRes-style residuals stack: [L, B, T, D] where L is the number of residual sources."""
    if L is None:
        raise ValueError("shape_LBTD requires the 'L' shape config key")
    return (L, B, T, D)


def shape_q_hq(B, T, H, D, HQ=None, **kw):
    """q with HQ query heads (GQA); k/v keep H heads. Used by NSA."""
    if HQ is None:
        raise ValueError("shape_q_hq requires the 'HQ' (query-head) shape config key")
    return (B, T, HQ, D)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

logsigmoid = F.logsigmoid


def sigmoid_transform(t):
    return t.sigmoid()


def logsigmoid_clamp(t):
    return F.logsigmoid(t).clamp_min(-5)


RWKV7_W_MIN = -0.6065306597126334


def rwkv7_w_transform(t):
    w = RWKV7_W_MIN * t.sigmoid()
    return w.clamp(min=RWKV7_W_MIN, max=-1e-6)


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
        name (str):
            Display and registry name, such as `chunk_gla`.
        import_path (str):
            Python module path, such as `fla.ops.gla`.
        inputs (dict[str, TensorSpec]):
            Mapping from function argument names to tensor specs.
        func_name (str, Optional):
            Function attribute name to import when it differs from `name`.
            Default: None.
        extra_kwargs (dict[str, Any], Optional):
            Constant keyword arguments passed to the op. Default: `{}`.
        output_is_tuple (bool):
            Whether the op returns a tuple whose first item is the tensor used for `.backward()`. Default: `True`.
        skip_backward (bool):
            Whether to skip forward-backward benchmark mode. Default: `False`.
        post_init (Callable, Optional):
            Callback invoked as `post_init(inputs, B=B, T=T, H=H, D=D, **kw)` for custom input mutation. Default: None.
        category (str):
            Grouping label used in reports. Default: `''`.
        dim_constraints (dict, Optional):
            Shape constraints, such as `{'D': [64, 128]}`. Shapes that do not match are skipped. Default: None.
        default_shapes (dict[str, dict[str, int]], Optional):
            Per-op shape configs used instead of the global `SHAPE_CONFIGS`.
            This is useful when the op's shape semantics differ from `B/T/H/D`,
            for example when AttnRes uses an extra `L` residual-source axis.
            Default: None.
        test_file (str, Optional):
            Path (relative to repo root) to the op's pytest file,
            used as the frozen correctness gate by `benchmarks/ops/verify.py`.
            When None, the gate is derived from `import_path`'s last segment (e.g. `fla.ops.gla` -> `tests/ops/test_gla.py`).
            Set this only when the derived path is wrong. Default: None.
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
    default_shapes: dict[str, dict[str, int]] | None = None
    test_file: str | None = None


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

# --- Simple qkv (no extra inputs) ---

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

# --- +elem gate (g=[B,T,H,D] with logsigmoid_clamp) ---

register_op(OpConfig(
    name='chunk_gla',
    import_path='fla.ops.gla',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTHD, transform=logsigmoid_clamp),
    },
    category='elem_gate',
))

# --- +beta (beta=[B,T,H] with sigmoid) ---

register_op(OpConfig(
    name='chunk_delta_rule',
    import_path='fla.ops.delta_rule',
    inputs={
        **_simple_qkv,
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    category='beta',
    test_file='tests/ops/test_delta.py',
))

# --- +gate + beta ---

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
    test_file='tests/ops/test_gdn.py',
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

# --- +head gate (g=[B,T,H] with logsigmoid) ---

register_op(OpConfig(
    name='chunk_simple_gla',
    import_path='fla.ops.simple_gla',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTH, transform=logsigmoid),
    },
    category='head_gate',
))

# --- RWKV ---


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
        'w': TensorSpec(shape_BTHD, transform=rwkv7_w_transform),
        'k': TensorSpec(shape_BTHD),
        'v': TensorSpec(shape_BTHD),
        'a': TensorSpec(shape_BTHD),
        'b': TensorSpec(shape_BTHD),
    },
    extra_kwargs={'safe_gate': True, 'chunk_size': 64},
    post_init=_rwkv7_post_init,
    category='rwkv',
))

# --- Comba ---

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

# --- HGRN (x, g only, no qkv) ---

register_op(OpConfig(
    name='fused_recurrent_hgrn',
    import_path='fla.ops.hgrn',
    inputs={
        'x': TensorSpec(shape_BTD),
        'g': TensorSpec(shape_BTD, transform=logsigmoid),
    },
    category='hgrn',
))

# --- Generalized delta rule (DPLR) ---

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
    test_file='tests/ops/test_dplr_delta.py',
))

# --- Lightning attention (needs layer_idx, num_layers) ---

register_op(OpConfig(
    name='chunk_lightning_attn',
    import_path='fla.ops.lightning_attn',
    inputs={**_simple_qkv},
    extra_kwargs={'layer_idx': 0, 'num_layers': 12},
    category='lightning',
))

# --- Attention baselines ---

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

# --- layer-axis residual aggregation (AttnRes, mHC, ...) ---
# These ops attend / aggregate over an `L` axis of stacked residual sources.
# Inputs and shape sweeps are shared so future ops (mHC etc.) can reuse them.

_layer_default_shapes = {
    'L8_B1_T8K_D2K':   {'L': 8,  'B': 1, 'T': 8192,  'H': 1, 'D': 2048},
    'L8_B1_T32K_D2K':  {'L': 8,  'B': 1, 'T': 32768, 'H': 1, 'D': 2048},
    'L10_B1_T8K_D4K':  {'L': 10, 'B': 1, 'T': 8192,  'H': 1, 'D': 4096},
    'L10_B1_T32K_D4K': {'L': 10, 'B': 1, 'T': 32768, 'H': 1, 'D': 4096},
    'L32_B1_T8K_D2K':  {'L': 64, 'B': 1, 'T': 8192,  'H': 1, 'D': 8192},
}


_attnres_inputs = {
    'query': TensorSpec(shape_D),
    'residuals': TensorSpec(shape_LBTD),
    'rms_weight': TensorSpec(shape_D),
}

register_op(OpConfig(
    name='fused_attnres',
    import_path='fla.ops.attnres',
    inputs=_attnres_inputs,
    output_is_tuple=False,
    default_shapes=_layer_default_shapes,
    category='fused_attnres',
))

register_op(OpConfig(
    name='naive_attnres',
    import_path='fla.ops.attnres',
    inputs=_attnres_inputs,
    output_is_tuple=False,
    default_shapes=_layer_default_shapes,
    category='naive_attnres',
))

# --- NSA (native sparse attention) — GQA + structured block selection ---
# q carries HQ query heads while k/v carry H kv heads (GQA; HQ/H a power of two
# and >= 16). block_indices is a causal random selection that must be built
# explicitly — the generic randn/randint input factory cannot produce a valid one.


def _nsa_post_init(inputs, B, T, H, D, HQ=None, S=16, block_size=64, **kw):
    # build block_indices [B, T, H, S]: for each query t, pick S of the causal
    # blocks (block i is selectable iff i <= t // block_size), -1-padded when
    # fewer than S exist, then sorted — mirroring tests/ops/test_nsa.py.
    device = inputs['q'].device
    NS = (T + block_size - 1) // block_size
    valid = (torch.arange(NS, device=device)[None, None, None, :]
             <= (torch.arange(T, device=device) // block_size)[None, :, None, None])
    scores = torch.rand(B, T, H, NS, device=device).masked_fill(~valid, float('-inf'))
    topv, topi = scores.topk(min(S, NS), dim=-1)
    block_indices = topi.masked_fill(topv == float('-inf'), -1).sort(-1)[0].to(torch.long)
    inputs['block_indices'] = block_indices.contiguous()
    inputs['block_counts'] = S
    inputs['block_size'] = block_size


_nsa_shapes = {
    'B1_T8K_H1_HQ32_D128_S16':  {'B': 1, 'T': 8192,  'H': 1, 'HQ': 32, 'D': 128, 'S': 16, 'block_size': 64},
    'B1_T16K_H1_HQ64_D128_S16': {'B': 1, 'T': 16384, 'H': 1, 'HQ': 64, 'D': 128, 'S': 16, 'block_size': 64},
    'B1_T32K_H1_HQ64_D256_S16': {'B': 1, 'T': 32768, 'H': 1, 'HQ': 64, 'D': 256, 'S': 16, 'block_size': 64},
    'B1_T64K_H1_HQ128_D256_S16': {'B': 1, 'T': 65536, 'H': 1, 'HQ': 128, 'D': 256, 'S': 16, 'block_size': 64},
    'B1_T128K_H1_HQ128_D256_S16': {'B': 1, 'T': 131072, 'H': 1, 'HQ': 128, 'D': 256, 'S': 16, 'block_size': 64},
}

register_op(OpConfig(
    name='parallel_nsa',
    import_path='fla.ops.nsa',
    inputs={
        'q': TensorSpec(shape_q_hq),
        'k': TensorSpec(shape_BTHD),
        'v': TensorSpec(shape_BTHD),
    },
    post_init=_nsa_post_init,
    output_is_tuple=False,
    default_shapes=_nsa_shapes,
    category='nsa',
    test_file='tests/ops/test_nsa.py',
))
