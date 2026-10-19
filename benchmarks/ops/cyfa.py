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
    init_cyfa_inputs,
    shape_bth,
    shape_bthd,
    shape_cyfa_readout,
    shape_d,
    sigmoid_transform,
)
from benchmarks.utils.ops.registry import OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator

OPERATORS = (
    OpConfig(
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
    ),
)

SCENARIOS = ('latency',)


def add_arguments(parser, scenario):
    add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    return run_operator(args=args, configs=OPERATORS)
