# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse
import os
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss


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
        x_names=['N', 'V', 'softcap'],
        x_vals=[(1024, 32000, None), (2048, 128256, None), (63, 131071, None),
                (1024, 32000, 3.0), (2047, 65539, 3.0), (7, 129, None)],
        line_arg='provider',
        line_vals=['triton', 'gluon', 'triton_bwd', 'gluon_bwd'],
        line_names=['Triton forward', 'Gluon forward', 'Triton forward+backward', 'Gluon forward+backward'],
        styles=[('blue', '-'), ('red', '-'), ('blue', '--'), ('red', '--')],
        ylabel='Execution time (us)',
        plot_name='CrossEntropyLogits',
        args={},
    ),
)
def benchmark_logits(N, V, softcap, provider):
    from fla.backends import _DISPATCH_DISABLED, GluonBackend
    from fla.utils import device

    if _DISPATCH_DISABLED or not GluonBackend.is_available():
        raise RuntimeError('This comparison requires NVIDIA Gluon and enabled backend dispatch')
    torch.manual_seed(42)
    logits = torch.randn(N, V, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(V, (N,), device=device)
    target[::7] = -100
    criterion = FusedCrossEntropyLoss(logit_softcapping=softcap)
    saved = {key: os.environ.get(key) for key in ('FLA_GLUON', 'FLA_CE_GLUON')}
    try:
        os.environ['FLA_GLUON'] = '0'
        os.environ['FLA_CE_GLUON'] = '1' if provider.startswith('gluon') else '0'

        def run():
            loss = criterion(logits, target)
            if provider.endswith('_bwd'):
                torch.autograd.grad(loss, logits)

        for _ in range(5):
            run()
        times = [triton.testing.do_bench_cudagraph(run, rep=100) * 1000 for _ in range(5)]
        return statistics.median(times), min(times), max(times)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logits-only', action='store_true', help='Compare Triton and Gluon CE without the linear projection')
    args = parser.parse_args()

    try:
        from runner import run_module_benchmark
    except ModuleNotFoundError:
        from benchmarks.modules.runner import run_module_benchmark

    run_module_benchmark(
        benchmark_logits if args.logits_only else benchmark,
        script_file=__file__,
        visualize=not args.logits_only,
    )
