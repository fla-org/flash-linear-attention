# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from benchmarks.utils.ops.inputs import (
    SIMPLE_QKV,
    TensorSpec,
    shape_bth,
    sigmoid_transform,
)
from benchmarks.utils.ops.registry import OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator

OPERATORS = (
    OpConfig(
        name='chunk_delta_rule',
        import_path='fla.ops.delta_rule',
        inputs={
            **SIMPLE_QKV,
            'beta': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
        },
        category='beta',
        test_file='tests/ops/test_delta.py',
    ),
)

SCENARIOS = ('latency',)


def add_arguments(parser, scenario):
    add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    return run_operator(args=args, configs=OPERATORS)
