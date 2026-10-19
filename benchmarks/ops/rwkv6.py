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
    TensorSpec,
    shape_bthd,
    shape_hd,
)
from benchmarks.utils.ops.registry import OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator

OPERATORS = (
    OpConfig(
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
    ),
)

SCENARIOS = ('latency',)


def add_arguments(parser, scenario):
    add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    return run_operator(args=args, configs=OPERATORS)
