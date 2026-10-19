# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from benchmarks.utils.ops.inputs import (
    TensorSpec,
    init_attnres_inputs,
    shape_d,
    shape_lbtd,
)
from benchmarks.utils.ops.registry import OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator

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

OPERATORS = (
    OpConfig(
        name='fused_attnres',
        import_path='fla.ops.attnres',
        inputs=_attnres_inputs,
        post_init=init_attnres_inputs,
        output_is_tuple=False,
        default_shapes=_layer_default_shapes,
        category='fused_attnres',
        backend_env={'gluon': 'FLA_ATTNRES_GLUON'},
    ),
    OpConfig(
        name='naive_attnres',
        import_path='fla.ops.attnres',
        inputs=_attnres_inputs,
        post_init=init_attnres_inputs,
        output_is_tuple=False,
        default_shapes=_layer_default_shapes,
        category='naive_attnres',
    ),
)

SCENARIOS = ('latency',)


def add_arguments(parser, scenario):
    add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    return run_operator(args=args, configs=OPERATORS)
