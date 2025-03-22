# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import triton

from fla.modules import GroupNorm, LayerNorm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['naive_ln',  'fused_ln', 'naive_gn', 'fused_gn',
                   'naive_ln_bwd',  'fused_ln_bwd', 'naive_gn_bwd', 'fused_gn_bwd'],
        # label name for the lines
        line_names=['naive_ln',  'fused_ln', 'naive_gn', 'fused_gn',
                    'naive_ln_bwd',  'fused_ln_bwd', 'naive_gn_bwd', 'fused_gn_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'),
                ('cyan', ':'), ('yellow', 'dotted'), ('cyan', '--'), ('cyan', '-'), ('black', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    requires_grad = True
    B, D = 16, 1024

    x = torch.randn(B * T, D, device=device, requires_grad=requires_grad, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider.startswith('naive_ln'):
        norm = nn.LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('fused_ln'):
        norm = LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('naive_gn'):
        norm = nn.GroupNorm(4, D).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('fused_gn'):
        norm = GroupNorm(4, D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('naive_ln_bwd'):
        norm = nn.LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    if provider.startswith('fused_ln_bwd'):
        norm = LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    if provider.startswith('naive_gn_bwd'):
        norm = nn.GroupNorm(4, D).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    if provider.startswith('fused_gn_bwd'):
        norm = GroupNorm(4, D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
