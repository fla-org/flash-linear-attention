# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from benchmarks.utils.ops.inputs import (
    TensorSpec,
    init_nsa_compression_inputs,
    init_nsa_inputs,
    init_nsa_windowed_inputs,
    shape_bthd,
    shape_nsa_compressed_k,
    shape_nsa_compressed_v,
    shape_nsa_gate,
    shape_nsa_v,
    shape_q_hq,
    sigmoid_transform,
)
from benchmarks.utils.ops.registry import OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator

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

_nsa_shapes = {
    'B1_T8K_H1_HQ32_D128_S16':  {'B': 1, 'T': 8192,  'H': 1, 'HQ': 32, 'D': 128, 'S': 16, 'block_size': 64},
    'B1_T16K_H1_HQ64_D128_S16': {'B': 1, 'T': 16384, 'H': 1, 'HQ': 64, 'D': 128, 'S': 16, 'block_size': 64},
    'B1_T32K_H1_HQ64_D256_S16': {'B': 1, 'T': 32768, 'H': 1, 'HQ': 64, 'D': 256, 'S': 16, 'block_size': 64},
    'B1_T64K_H1_HQ128_D256_S16': {'B': 1, 'T': 65536, 'H': 1, 'HQ': 128, 'D': 256, 'S': 16, 'block_size': 64},
    'B1_T128K_H1_HQ128_D256_S16': {'B': 1, 'T': 131072, 'H': 1, 'HQ': 128, 'D': 256, 'S': 16, 'block_size': 64},
}

OPERATORS = (
    OpConfig(
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
    ),
    OpConfig(
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
    ),
    OpConfig(
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
    ),
)

SCENARIOS = ('latency',)


def add_arguments(parser, scenario):
    add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    return run_operator(args=args, configs=OPERATORS)
