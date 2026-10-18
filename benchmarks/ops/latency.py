# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Forward and forward/backward latency of registered operators."""

import json

from benchmarks.utils.ops.registry import SHAPE_CONFIGS, list_ops
from benchmarks.utils.ops.runner import benchmark_op


def add_arguments(parser):
    parser.add_argument('--op', nargs='+', required=True, choices=[*list_ops(), 'all'])
    parser.add_argument('--backend', help='Registered backend override, such as triton or gluon')
    parser.add_argument('--custom-shapes', help='JSON object keyed by shape name, with B/T/H/D and optional operator axes')
    parser.add_argument('--modes', nargs='+', choices=['fwd', 'fwdbwd'], default=['fwd', 'fwdbwd'])


def run(args):
    op_names = list_ops() if args.op == ['all'] else args.op
    shapes = json.loads(args.custom_shapes) if args.custom_shapes else SHAPE_CONFIGS
    results = []
    for name in op_names:
        results.extend(benchmark_op(op_name=name, shapes=shapes, modes=args.modes, backend=args.backend))
    return results
