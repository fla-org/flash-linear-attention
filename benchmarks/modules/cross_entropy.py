# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

from benchmarks.utils.perf_report import add_module_arguments, run_module_benchmark
from fla.backends import GluonBackend
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss

LOGITS_PROVIDERS = ['triton', 'triton_bwd']
if GluonBackend.is_available() and os.environ.get('FLA_DISABLE_BACKEND_DISPATCH') != '1':
    LOGITS_PROVIDERS.extend(['gluon', 'gluon_bwd'])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['naive',  'fused', 'fused_linear', 'naive_bwd',  'fused_bwd', 'fused_linear_bwd'],
        # label name for the lines
        line_names=['naive',  'fused', 'fused_linear', 'naive_bwd',  'fused_bwd', 'fused_linear_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'),
                ('cyan', ':'), ('yellow', 'dotted'), ('cyan', '--'), ('cyan', '-'), ('black', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    ),
)
def benchmark(T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    requires_grad = True
    B, H, V = 4, 4096, 120000

    x = torch.randn(B * T, H, device=device, requires_grad=requires_grad, dtype=dtype)
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.int64)
    w = torch.randn(V, H, device=device, requires_grad=requires_grad, dtype=dtype)
    b = torch.randn(V, device=device, requires_grad=requires_grad, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'naive':
        criterion = nn.CrossEntropyLoss()
        results = triton.testing.do_bench(lambda: criterion(F.linear(x, w, b), target), quantiles=quantiles)
    elif provider == 'naive_bwd':
        criterion = nn.CrossEntropyLoss()
        results = triton.testing.do_bench(lambda: criterion(F.linear(x, w, b), target).backward(), quantiles=quantiles)
    elif provider == 'fused':
        criterion = FusedCrossEntropyLoss()
        results = triton.testing.do_bench(lambda: criterion(F.linear(x, w, b), target), quantiles=quantiles)
    elif provider == 'fused_bwd':
        criterion = FusedCrossEntropyLoss()
        results = triton.testing.do_bench(lambda: criterion(F.linear(x, w, b), target).backward(), quantiles=quantiles)
    elif provider == 'fused_linear':
        criterion = FusedLinearCrossEntropyLoss()
        results = triton.testing.do_bench(lambda: criterion(x, target, w, b), quantiles=quantiles)
    elif provider == 'fused_linear_bwd':
        criterion = FusedLinearCrossEntropyLoss()
        results = triton.testing.do_bench(lambda: criterion(x, target, w, b).backward(), quantiles=quantiles)
    return results


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N', 'V', 'label_smoothing', 'logit_softcapping', 'dtype'],
        x_vals=[
            (1024, 32000, 0.0, None, torch.bfloat16),
            (2048, 128256, 0.0, None, torch.bfloat16),
            (63, 131071, 0.0, None, torch.bfloat16),
            (1024, 32000, 0.0, 3.0, torch.bfloat16),
            (2047, 65539, 0.0, 3.0, torch.bfloat16),
            (7, 129, 0.0, None, torch.bfloat16),
            (7, 8193, 0.0, None, torch.float16),
            (1024, 16384, 0.0, None, torch.bfloat16),
            (1024, 16384, 0.0, None, torch.float32),
            (1025, 32000, 0.1, None, torch.bfloat16),
            (511, 32003, 0.1, 3.0, torch.float32),
        ],
        line_arg='provider',
        line_vals=LOGITS_PROVIDERS,
        line_names=LOGITS_PROVIDERS,
        styles=[('blue', '-'), ('blue', '--'), ('red', '-'), ('red', '--')][:len(LOGITS_PROVIDERS)],
        ylabel='Execution Time (ms)',
        plot_name='CrossEntropyLogits',
        args={},
    ),
)
def benchmark_logits(
    N: int,
    V: int,
    label_smoothing: float,
    logit_softcapping: float | None,
    dtype: torch.dtype,
    provider: str,
):
    from fla.utils import device

    torch.manual_seed(42)
    os.environ['FLA_GLUON'] = '0'
    os.environ['FLA_CE_GLUON'] = '1' if provider.startswith('gluon') else '0'
    logits = torch.randn(N, V, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(V, (N,), device=device)
    target[::7] = -100
    criterion = FusedCrossEntropyLoss(label_smoothing=label_smoothing, logit_softcapping=logit_softcapping)
    quantiles = [0.5, 0.2, 0.8]
    if provider.endswith('_bwd'):
        return triton.testing.do_bench(
            lambda: criterion(input=logits, target=target).backward(),
            grad_to_none=[logits],
            quantiles=quantiles,
        )
    return triton.testing.do_bench(lambda: criterion(input=logits, target=target), quantiles=quantiles)


def add_arguments(parser):
    add_module_arguments(parser)
    parser.add_argument('--logits-only', action='store_true', help='Compare cross entropy without the linear projection')


def run(args):
    return run_module_benchmark(
        mark=benchmark_logits if args.logits_only else benchmark,
        save_dir=args.save_dir,
        plot=args.plot,
    )
