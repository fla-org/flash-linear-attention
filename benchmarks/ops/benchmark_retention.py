# -*- coding: utf-8 -*-

import os

import torch
import triton
from flash_attn import flash_attn_func

from fla.ops.retention import chunk_retention, parallel_retention


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['chunk', 'parallel', 'flash', 'chunk_bwd', 'parallel_bwd', 'flash_bwd'],
        # label name for the lines
        line_names=['chunk_fwd', 'parallel_fwd', 'flash_fwd', 'chunk_fwdbwd', 'parallel_fwdbwd', 'flash_fwdbwd'],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('green', 'dotted'), ('blue', 'dotted'), ('red', 'dotted')],
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
    B, H, D = 4, 8, 256
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if provider == 'flash' or provider == 'flash_bwd':
        q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    else:
        q = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    do = torch.ones_like(q, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'chunk':
        results = triton.testing.do_bench(lambda: chunk_retention(q, k, v), quantiles=quantiles)
    elif provider == 'parallel':
        results = triton.testing.do_bench(lambda: parallel_retention(q, k, v), quantiles=quantiles)
    elif provider == 'flash':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True), quantiles=quantiles)
    elif provider == 'chunk_bwd':
        results = triton.testing.do_bench(lambda: chunk_retention(q, k, v)[0].backward(do), quantiles=quantiles)
    elif provider == 'parallel_bwd':
        results = triton.testing.do_bench(lambda: parallel_retention(q, k, v)[0].backward(do), quantiles=quantiles)
    elif provider == 'flash_bwd':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True).backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, save_path='.')
