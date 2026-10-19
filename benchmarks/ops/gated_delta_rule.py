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
    sigmoid_transform,
)
from benchmarks.utils.ops.registry import OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator

OPERATORS = (
    OpConfig(
        name='chunk_gdn',
        import_path='fla.ops.gated_delta_rule',
        inputs={
            **SIMPLE_QKV,
            'g': TensorSpec(shape_fn=shape_bth, transform=logsigmoid),
            'beta': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
        },
        func_name='chunk_gated_delta_rule',
        extra_kwargs={'use_qk_l2norm_in_kernel': True},
        category='gate_beta',
        test_file='tests/ops/test_gdn.py',
    ),
)

SCENARIOS = ('latency', 'cp')


def add_arguments(parser, scenario):
    if scenario == 'cp':
        from benchmarks.utils.distributed import add_cp_arguments
        add_cp_arguments(parser)
    else:
        add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    if args.scenario == 'cp':
        from benchmarks.utils.distributed import run_distributed
        from benchmarks.utils.ops.distributed import benchmark_ops
        args.kda = False
        return run_distributed(args=args, measure=benchmark_ops)
    return run_operator(args=args, configs=OPERATORS)
