# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Registered operators, arguments, and default workloads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from torch.nn.functional import logsigmoid

from benchmarks.ops.utils.inputs import (
    TensorSpec,
    init_attnres_inputs,
    init_cyfa_inputs,
    init_nsa_compression_inputs,
    init_nsa_inputs,
    init_nsa_windowed_inputs,
    init_rwkv7_inputs,
    logsigmoid_clamp,
    rwkv7_w_transform,
    shape_btd,
    shape_bth,
    shape_bthd,
    shape_cyfa_readout,
    shape_d,
    shape_h,
    shape_hd,
    shape_lbtd,
    shape_nsa_compressed_k,
    shape_nsa_compressed_v,
    shape_nsa_gate,
    shape_nsa_v,
    shape_q_hq,
    sigmoid_transform,
)


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
            Default: `None`.
        extra_kwargs (dict[str, Any], Optional):
            Constant keyword arguments passed to the op. Default: `{}`.
        output_is_tuple (bool):
            Whether the op returns a tuple whose first item is the tensor used for `.backward()`. Default: `True`.
        skip_backward (bool):
            Whether to skip forward-backward benchmark mode. Default: `False`.
        post_init (Callable, Optional):
            Callback invoked as `post_init(inputs, B=B, T=T, H=H, D=D, **kw)` for custom input mutation. Default: `None`.
        category (str):
            Grouping label used in reports. Default: `''`.
        dim_constraints (dict, Optional):
            Shape constraints, such as `{'D': [64, 128]}`. Shapes that do not match are skipped. Default: `None`.
        default_shapes (dict[str, dict[str, int]], Optional):
            Per-op shape configs used instead of the global `SHAPE_CONFIGS`.
            This is useful when the op's shape semantics differ from `B/T/H/D`,
            for example when AttnRes uses an extra `L` residual-source axis.
            Default: `None`.
        test_file (str, Optional):
            Path (relative to repo root) to the op's pytest file,
            used as the frozen correctness gate by `benchmarks/verify.py`.
            When None, the gate is derived from `import_path`'s last segment (e.g. `fla.ops.gla` -> `tests/ops/test_gla.py`).
            Set this only when the derived path is wrong. Default: `None`.
        backend_env (dict[str, str], Optional):
            Maps a `--backend <name>` value to the environment variable that enables that backend's
            dispatch (e.g. `{'gluon': 'FLA_ATTNRES_GLUON'}`). The runner sets it before launching the
            op; ops selected purely by a runtime verifier need no entry. Default: `None`.
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
    backend_env: dict[str, str] | None = None


_REGISTRY: dict[str, OpConfig] = {}


def register_op(config: OpConfig) -> None:
    _REGISTRY[config.name] = config


def get_op(name: str) -> OpConfig:
    if name not in _REGISTRY:
        raise KeyError(f"Op '{name}' not registered. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_ops() -> list[str]:
    return sorted(_REGISTRY.keys())


SHAPE_CONFIGS = {
    'B1_T8192_H96_D128':  {'B': 1,  'T': 8192,  'H': 96, 'D': 128},
    'B2_T16384_H16_D128': {'B': 2,  'T': 16384, 'H': 16, 'D': 128},
    'B4_T2048_H16_D128':  {'B': 4,  'T': 2048,  'H': 16, 'D': 128},
    'B4_T4096_H64_D128':  {'B': 4,  'T': 4096,  'H': 64, 'D': 128},
    'B8_T2048_H32_D256':  {'B': 8,  'T': 2048,  'H': 32, 'D': 256},
    'B8_T1024_H8_D64':    {'B': 8,  'T': 1024,  'H': 8,  'D': 64},
}

register_op(OpConfig(
    name='chunk_cyfa',
    import_path='fla.ops.cyfa',
    inputs={
        'q': TensorSpec(shape_fn=shape_bthd),
        'k': TensorSpec(shape_fn=shape_bthd),
        'v': TensorSpec(shape_fn=shape_bthd),
        'g': TensorSpec(shape_fn=shape_bth, dtype='float32', transform=logsigmoid),
        'delta': TensorSpec(shape_fn=shape_bth, dtype='float32', transform=sigmoid_transform),
        'beta': TensorSpec(shape_fn=shape_bth, dtype='float32', transform=sigmoid_transform),
        'readout': TensorSpec(shape_fn=shape_cyfa_readout, dtype='float32'),
        'q_norm_weight': TensorSpec(shape_fn=shape_d, dtype='float32'),
        'k_norm_weight': TensorSpec(shape_fn=shape_d, dtype='float32'),
    },
    extra_kwargs={'q_norm_eps': 1e-6, 'k_norm_eps': 1e-6, 'output_final_state': True},
    post_init=init_cyfa_inputs,
))

_simple_qkv = {
    'q': TensorSpec(shape_fn=shape_bthd),
    'k': TensorSpec(shape_fn=shape_bthd),
    'v': TensorSpec(shape_fn=shape_bthd),
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

register_op(OpConfig(
    name='chunk_gla',
    import_path='fla.ops.gla',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid_clamp),
    },
    category='elem_gate',
))

register_op(OpConfig(
    name='chunk_delta_rule',
    import_path='fla.ops.delta_rule',
    inputs={
        **_simple_qkv,
        'beta': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
    },
    category='beta',
    test_file='tests/ops/test_delta.py',
))

register_op(OpConfig(
    name='chunk_gdn',
    import_path='fla.ops.gated_delta_rule',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_fn=shape_bth, transform=logsigmoid),
        'beta': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
    },
    func_name='chunk_gated_delta_rule',
    extra_kwargs={'use_qk_l2norm_in_kernel': True},
    category='gate_beta',
    test_file='tests/ops/test_gdn.py',
))

_gdn2_default_shapes = {
    **SHAPE_CONFIGS,
    'B1_T32768_H1_D32': {'B': 1, 'T': 32768, 'H': 1, 'D': 32},
}

register_op(OpConfig(
    name='chunk_gdn2',
    import_path='fla.ops.gdn2',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid),
        'b': TensorSpec(shape_fn=shape_bthd, transform=sigmoid_transform),
        'w': TensorSpec(shape_fn=shape_bthd, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True},
    category='gate_beta',
    default_shapes=_gdn2_default_shapes,
    test_file='tests/ops/test_gdn2.py',
))

register_op(OpConfig(
    name='chunk_kda',
    import_path='fla.ops.kda',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid),
        'beta': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True, 'safe_gate': True, 'lower_bound': -5},
    category='gate_beta',
))

register_op(OpConfig(
    name='fused_recurrent_gdn2',
    import_path='fla.ops.gdn2',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid),
        'b': TensorSpec(shape_fn=shape_bthd, transform=sigmoid_transform),
        'w': TensorSpec(shape_fn=shape_bthd, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True, 'output_final_state': True},
    skip_backward=True,
    category='gdn2',
))

register_op(OpConfig(
    name='chunk_precond_gdn',
    import_path='fla.ops.precond_gated_delta_rule',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_fn=shape_bth, transform=logsigmoid),
        'beta': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
        'g_atk': TensorSpec(shape_fn=shape_bth, transform=logsigmoid),
        'beta_atk': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
        'log_atk_scale': TensorSpec(shape_fn=shape_h, dtype='float32'),
    },
    func_name='chunk_precond_gated_delta_rule',
    extra_kwargs={'use_qk_l2norm_in_kernel': True, 'x': 1.5},
    category='gate_beta',
    test_file='tests/ops/test_precond_gated_delta.py',
))

register_op(OpConfig(
    name='chunk_precond_kda',
    import_path='fla.ops.precond_kda',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid),
        'beta': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
        'g_atk': TensorSpec(shape_fn=shape_bth, transform=logsigmoid),
        'beta_atk': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
        'log_atk_scale': TensorSpec(shape_fn=shape_h, dtype='float32'),
    },
    extra_kwargs={'x': 1.5},
    category='gate_beta',
    test_file='tests/ops/test_precond_kda.py',
))

register_op(OpConfig(
    name='chunk_simple_gla',
    import_path='fla.ops.simple_gla',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_fn=shape_bth, transform=logsigmoid),
    },
    category='head_gate',
))

register_op(OpConfig(
    name='chunk_rwkv6',
    import_path='fla.ops.rwkv6',
    inputs={
        'r': TensorSpec(shape_fn=shape_bthd),
        'k': TensorSpec(shape_fn=shape_bthd),
        'v': TensorSpec(shape_fn=shape_bthd),
        'w': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid),
        'u': TensorSpec(shape_fn=shape_hd, requires_grad=False),
    },
    category='rwkv',
))

register_op(OpConfig(
    name='chunk_rwkv7',
    import_path='fla.ops.rwkv7',
    inputs={
        'r': TensorSpec(shape_fn=shape_bthd),
        'w': TensorSpec(shape_fn=shape_bthd, transform=rwkv7_w_transform),
        'k': TensorSpec(shape_fn=shape_bthd),
        'v': TensorSpec(shape_fn=shape_bthd),
        'a': TensorSpec(shape_fn=shape_bthd),
        'b': TensorSpec(shape_fn=shape_bthd),
    },
    extra_kwargs={'safe_gate': True, 'chunk_size': 64},
    post_init=init_rwkv7_inputs,
    category='rwkv',
))

register_op(OpConfig(
    name='chunk_comba',
    import_path='fla.ops.comba',
    inputs={
        **_simple_qkv,
        'p': TensorSpec(shape_fn=shape_bthd),
        'g': TensorSpec(shape_fn=shape_bth, transform=logsigmoid),
        'beta': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True},
    category='comba',
))

register_op(OpConfig(
    name='fused_recurrent_hgrn',
    import_path='fla.ops.hgrn',
    inputs={
        'x': TensorSpec(shape_fn=shape_btd),
        'g': TensorSpec(shape_fn=shape_btd, transform=logsigmoid),
    },
    category='hgrn',
))

register_op(OpConfig(
    name='chunk_dplr_delta_rule',
    import_path='fla.ops.generalized_delta_rule',
    inputs={
        **_simple_qkv,
        'a': TensorSpec(shape_fn=shape_bthd),
        'b': TensorSpec(shape_fn=shape_bthd),
        'gk': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid),
    },
    category='gen_delta',
    test_file='tests/ops/test_dplr_delta.py',
))

register_op(OpConfig(
    name='chunk_lightning_attn',
    import_path='fla.ops.lightning_attn',
    inputs={**_simple_qkv},
    extra_kwargs={'layer_idx': 0, 'num_layers': 12},
    category='lightning',
))

register_op(OpConfig(
    name='parallel_attn',
    import_path='fla.ops.attn',
    inputs={**_simple_qkv},
    output_is_tuple=False,
    category='attn',
))

register_op(OpConfig(
    name='parallel_stickbreaking_attn',
    import_path='fla.ops.stickbreaking_attn',
    inputs={**_simple_qkv},
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

_layer_default_shapes = {
    'L8_B1_T8K_D2K':   {'L': 8,  'B': 1, 'T': 8192,  'H': 1, 'D': 2048},
    'L8_B1_T32K_D2K':  {'L': 8,  'B': 1, 'T': 32768, 'H': 1, 'D': 2048},
    'L10_B1_T8K_D4K':  {'L': 10, 'B': 1, 'T': 8192,  'H': 1, 'D': 4096},
    'L10_B1_T32K_D4K': {'L': 10, 'B': 1, 'T': 32768, 'H': 1, 'D': 4096},
    'L32_B1_T8K_D2K':  {'L': 64, 'B': 1, 'T': 8192,  'H': 1, 'D': 8192},
}

_attnres_inputs = {
    'query': TensorSpec(shape_fn=shape_d),
    'residuals': TensorSpec(shape_fn=shape_lbtd),
    'rms_weight': TensorSpec(shape_fn=shape_d),
}

register_op(OpConfig(
    name='fused_attnres',
    import_path='fla.ops.attnres',
    inputs=_attnres_inputs,
    post_init=init_attnres_inputs,
    output_is_tuple=False,
    default_shapes=_layer_default_shapes,
    category='fused_attnres',
    backend_env={'gluon': 'FLA_ATTNRES_GLUON'},
))

register_op(OpConfig(
    name='naive_attnres',
    import_path='fla.ops.attnres',
    inputs=_attnres_inputs,
    post_init=init_attnres_inputs,
    output_is_tuple=False,
    default_shapes=_layer_default_shapes,
    category='naive_attnres',
))

# NSA uses HQ query heads; HQ/H must be a power of two and at least 16.
_nsa_compression_bq_shapes = {
    'B1_T8K_H4_HQ64_K32_V32': {
        'B': 1, 'T': 8192, 'H': 4, 'HQ': 64, 'D': 32, 'V': 32, 'S': 16, 'block_size': 64,
    },
    'B1_T16K_H4_HQ64_K32_V32': {
        'B': 1, 'T': 16384, 'H': 4, 'HQ': 64, 'D': 32, 'V': 32, 'S': 16, 'block_size': 64,
    },
    'B1_T32K_H4_HQ64_K32_V32': {
        'B': 1, 'T': 32768, 'H': 4, 'HQ': 64, 'D': 32, 'V': 32, 'S': 16, 'block_size': 64,
    },
    'B1_T16K_H4_HQ64_K64_V128': {
        'B': 1, 'T': 16384, 'H': 4, 'HQ': 64, 'D': 64, 'V': 128, 'S': 16, 'block_size': 64,
    },
}

register_op(OpConfig(
    name='parallel_nsa_compression',
    import_path='fla.ops.nsa.compression',
    inputs={
        'q': TensorSpec(shape_fn=shape_q_hq),
        'k': TensorSpec(shape_fn=shape_nsa_compressed_k),
        'v': TensorSpec(shape_fn=shape_nsa_compressed_v),
    },
    post_init=init_nsa_compression_inputs,
    output_is_tuple=True,
    default_shapes=_nsa_compression_bq_shapes,
    category='nsa',
    test_file='tests/ops/test_nsa.py',
))

register_op(OpConfig(
    name='parallel_nsa_windowed',
    import_path='fla.ops.nsa',
    func_name='parallel_nsa',
    inputs={
        'q': TensorSpec(shape_fn=shape_q_hq),
        'k': TensorSpec(shape_fn=shape_bthd),
        'v': TensorSpec(shape_fn=shape_nsa_v),
        'g_cmp': TensorSpec(shape_fn=shape_nsa_gate, transform=sigmoid_transform),
        'g_slc': TensorSpec(shape_fn=shape_nsa_gate, transform=sigmoid_transform),
        'g_swa': TensorSpec(shape_fn=shape_nsa_gate, transform=sigmoid_transform),
    },
    post_init=init_nsa_windowed_inputs,
    output_is_tuple=False,
    default_shapes=_nsa_compression_bq_shapes,
    category='nsa',
    test_file='tests/ops/test_nsa.py',
))

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
        'q': TensorSpec(shape_fn=shape_q_hq),
        'k': TensorSpec(shape_fn=shape_bthd),
        'v': TensorSpec(shape_fn=shape_bthd),
    },
    post_init=init_nsa_inputs,
    output_is_tuple=False,
    default_shapes=_nsa_shapes,
    category='nsa',
    test_file='tests/ops/test_nsa.py',
))
