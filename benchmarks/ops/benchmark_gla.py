# -*- coding: utf-8 -*-

import torch
import triton
from torch.nn import functional as F

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.ops.retention import chunk_retention, parallel_retention
from fla.ops.retention.naive import naive_retention


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['fused_chunk_gla', 'recurrent_gla',  'chunk_gla', 'chunk_retention',
                   'fused_chunk_gla_bwd',  'recurrent_gla_bwd', 'chunk_gla_bwd', 'chunk_retention_bwd'],
        # label name for the lines
        line_names=['fused_chunk_gla', 'recurrent_gla',  'chunk_gla',  'chunk_retention',
                    'fused_chunk_gla_bwd', 'recurrent_gla_bwd', 'chunk_gla_bwd',  'chunk_retention_bwd'],
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
    B, H, D = 16, 8, 128

    q = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, H, T, D, device=device, dtype=dtype)).clamp_min(-5).requires_grad_(requires_grad)
    v = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)

    do = torch.ones_like(q, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'torch':
        if T > 2048:
            return results
        results = triton.testing.do_bench(lambda: naive_retention(q, k, v), quantiles=quantiles)
    elif provider == 'recurrent_gla':
        results = triton.testing.do_bench(lambda: fused_recurrent_gla(q, k, v, g), quantiles=quantiles)
    elif provider == 'fused_chunk_gla':
        results = triton.testing.do_bench(lambda: fused_chunk_gla(q, k, v, g), quantiles=quantiles)
    elif provider == 'chunk_retention':
        results = triton.testing.do_bench(lambda: chunk_retention(q, k, v), quantiles=quantiles)
    elif provider == 'chunk_gla':
        results = triton.testing.do_bench(lambda: chunk_gla(q, k, v, g), quantiles=quantiles)
    elif provider == 'parallel':
        results = triton.testing.do_bench(lambda: parallel_retention(q, k, v), quantiles=quantiles)
    elif provider == 'torch_bwd':
        if T > 2048:
            return results
    elif provider == 'chunk_retention_bwd':
        results = triton.testing.do_bench(lambda: chunk_retention(q, k, v)[0].backward(do), quantiles=quantiles)
    elif provider == 'recurrent_gla_bwd':
        results = triton.testing.do_bench(lambda: fused_recurrent_gla(q, k, v, g)[0].backward(do), quantiles=quantiles)
    elif provider == 'fused_chunk_gla_bwd':
        results = triton.testing.do_bench(lambda: fused_chunk_gla(q, k, v, g)[0].backward(do), quantiles=quantiles)
    elif provider == 'chunk_gla_bwd':
        results = triton.testing.do_bench(lambda: chunk_gla(q, k, v, g)[0].backward(do), quantiles=quantiles)
    elif provider == 'parallel_bwd':
        results = triton.testing.do_bench(lambda: parallel_retention(q, k, v)[0].backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
