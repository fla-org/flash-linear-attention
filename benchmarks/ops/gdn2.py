# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from torch.nn.functional import (
    logsigmoid,
)

from benchmarks.utils.ops.inputs import (
    SIMPLE_QKV,
    TensorSpec,
    shape_bthd,
    sigmoid_transform,
)
from benchmarks.utils.ops.registry import SHAPE_CONFIGS, OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator

_gdn2_default_shapes = {
    **SHAPE_CONFIGS,
    'B1_T32768_H1_D32': {'B': 1, 'T': 32768, 'H': 1, 'D': 32},
}

OPERATORS = (
    OpConfig(
        name='chunk_gdn2',
        import_path='fla.ops.gdn2',
        inputs={
            **SIMPLE_QKV,
            'g': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid),
            'b': TensorSpec(shape_fn=shape_bthd, transform=sigmoid_transform),
            'w': TensorSpec(shape_fn=shape_bthd, transform=sigmoid_transform),
        },
        extra_kwargs={'use_qk_l2norm_in_kernel': True},
        category='gate_beta',
        default_shapes=_gdn2_default_shapes,
        test_file='tests/ops/test_gdn2.py',
    ),
    OpConfig(
        name='fused_recurrent_gdn2',
        import_path='fla.ops.gdn2',
        inputs={
            **SIMPLE_QKV,
            'g': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid),
            'b': TensorSpec(shape_fn=shape_bthd, transform=sigmoid_transform),
            'w': TensorSpec(shape_fn=shape_bthd, transform=sigmoid_transform),
        },
        extra_kwargs={'use_qk_l2norm_in_kernel': True, 'output_final_state': True},
        skip_backward=True,
        category='gdn2',
    ),
)

SCENARIOS = ('latency',)


def add_arguments(parser, scenario):
    add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    return run_operator(args=args, configs=OPERATORS)
