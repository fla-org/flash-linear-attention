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
    shape_btd,
)
from benchmarks.utils.ops.registry import OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator

OPERATORS = (
    OpConfig(
        name='fused_recurrent_hgrn',
        import_path='fla.ops.hgrn',
        inputs={
            'x': TensorSpec(shape_fn=shape_btd),
            'g': TensorSpec(shape_fn=shape_btd, transform=logsigmoid),
        },
        category='hgrn',
    ),
)

SCENARIOS = ('latency',)


def add_arguments(parser, scenario):
    add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    return run_operator(args=args, configs=OPERATORS)
