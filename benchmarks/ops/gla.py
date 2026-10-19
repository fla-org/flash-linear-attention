# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from benchmarks.utils.ops.inputs import (
    SIMPLE_QKV,
    TensorSpec,
    logsigmoid_clamp,
    shape_bthd,
)
from benchmarks.utils.ops.registry import OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator

OPERATORS = (
    OpConfig(
        name='chunk_gla',
        import_path='fla.ops.gla',
        inputs={
            **SIMPLE_QKV,
            'g': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid_clamp),
        },
        category='elem_gate',
    ),
)

SCENARIOS = ('latency',)


def add_arguments(parser, scenario):
    add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    return run_operator(args=args, configs=OPERATORS)
