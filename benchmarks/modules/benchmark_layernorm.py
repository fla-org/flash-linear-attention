# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

from fla.modules import FusedLayerNormGated, FusedRMSNormGated, GroupNorm, LayerNorm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=[
            'naive_ln', 'fused_ln', 'naive_gn', 'fused_gn',
            'naive_ln_bwd', 'fused_ln_bwd', 'naive_gn_bwd', 'fused_gn_bwd',
            'naive_ln_gate', 'fused_ln_gate', 'naive_rms_gate', 'fused_rms_gate',
            'naive_ln_gate_bwd', 'fused_ln_gate_bwd', 'naive_rms_gate_bwd', 'fused_rms_gate_bwd',
        ],
        # label name for the lines
        line_names=[
            'naive_ln', 'fused_ln', 'naive_gn', 'fused_gn',
            'naive_ln_bwd', 'fused_ln_bwd', 'naive_gn_bwd', 'fused_gn_bwd',
            'naive_ln_gate', 'fused_ln_gate', 'naive_rms_gate', 'fused_rms_gate',
            'naive_ln_gate_bwd', 'fused_ln_gate_bwd', 'naive_rms_gate_bwd', 'fused_rms_gate_bwd',
        ],
        # line styles
        styles=[
            ('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':'),
            ('yellow', 'dotted'), ('cyan', '--'), ('cyan', '-'), ('black', ':'),
            ('magenta', '-'), ('orange', '--'), ('purple', '-.'), ('brown', ':'),
            ('pink', 'dotted'), ('olive', '--'), ('navy', '-'), ('gray', ':'),
        ],
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
    B, D = 16, 1024
    activation = 'silu'

    x = torch.randn(B * T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    g = torch.randn(B * T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    do = torch.randn_like(x)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    if provider == 'naive_ln':
        norm = nn.LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    elif provider == 'fused_ln':
        norm = LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    elif provider == 'naive_gn':
        norm = nn.GroupNorm(4, D).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    elif provider == 'fused_gn':
        norm = GroupNorm(4, D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    elif provider == 'naive_ln_bwd':
        norm = nn.LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    elif provider == 'fused_ln_bwd':
        norm = LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    elif provider == 'naive_gn_bwd':
        norm = nn.GroupNorm(4, D).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    elif provider == 'fused_gn_bwd':
        norm = GroupNorm(4, D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    elif provider.startswith('naive_ln_gate'):
        norm = nn.LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        if provider.endswith('bwd'):
            results = triton.testing.do_bench(
                lambda: (norm(x) * F.silu(g)).backward(do),
                quantiles=quantiles,
            )
        else:
            results = triton.testing.do_bench(lambda: norm(x) * F.silu(g), quantiles=quantiles)
    elif provider.startswith('fused_ln_gate'):
        norm = FusedLayerNormGated(
            D, elementwise_affine=True, bias=True, activation=activation,
        ).to(device=device, dtype=dtype)
        if provider.endswith('bwd'):
            results = triton.testing.do_bench(
                lambda: norm(x, g).backward(do),
                quantiles=quantiles,
            )
        else:
            results = triton.testing.do_bench(lambda: norm(x, g), quantiles=quantiles)
    elif provider.startswith('naive_rms_gate'):
        norm = nn.RMSNorm(D).to(device=device, dtype=dtype)
        if provider.endswith('bwd'):
            results = triton.testing.do_bench(
                lambda: (norm(x) * F.silu(g)).backward(do),
                quantiles=quantiles,
            )
        else:
            results = triton.testing.do_bench(lambda: norm(x) * F.silu(g), quantiles=quantiles)
    elif provider.startswith('fused_rms_gate'):
        norm = FusedRMSNormGated(D, activation=activation).to(device=device, dtype=dtype)
        if provider.endswith('bwd'):
            results = triton.testing.do_bench(
                lambda: norm(x, g).backward(do),
                quantiles=quantiles,
            )
        else:
            results = triton.testing.do_bench(lambda: norm(x, g), quantiles=quantiles)
    return results


if __name__ == '__main__':
    try:
        from runner import run_module_benchmark
    except ModuleNotFoundError:
        from benchmarks.modules.runner import run_module_benchmark

    run_module_benchmark(benchmark, script_file=__file__)
