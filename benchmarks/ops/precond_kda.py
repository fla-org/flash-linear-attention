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
    shape_bth,
    shape_bthd,
    shape_h,
    sigmoid_transform,
)
from benchmarks.utils.ops.registry import OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator

OPERATORS = (
    OpConfig(
        name='chunk_precond_kda',
        import_path='fla.ops.precond_kda',
        inputs={
            **SIMPLE_QKV,
            'g': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid),
            'beta': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
            'g_atk': TensorSpec(shape_fn=shape_bth, transform=logsigmoid),
            'beta_atk': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
            'log_atk_scale': TensorSpec(shape_fn=shape_h, dtype='float32'),
        },
        extra_kwargs={'x': 1.5},
        category='gate_beta',
        test_file='tests/ops/test_precond_kda.py',
    ),
)

SCENARIOS = ('latency',)


def add_arguments(parser, scenario):
    add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    return run_operator(args=args, configs=OPERATORS)
