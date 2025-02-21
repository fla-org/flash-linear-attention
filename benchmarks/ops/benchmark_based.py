# -*- coding: utf-8 -*-

import torch
import triton

from fla.ops.based import fused_chunk_based, parallel_based
from fla.ops.based.naive import naive_chunk_based, naive_parallel_based

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except Exception:
    HAS_FLASH = False


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(3, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        line_vals=['fused_chunk', 'torch', 'parallel', 'parallel_chunk', 'fused_chunk_bwd', 'torch_bwd',
                   'parallel_bwd', 'parallel_chunk_bwd'] + (['flash', 'flash_bwd'] if HAS_FLASH else []),
        # label name for the lines
        line_names=['fused_chunk_fwd', 'torch_fwd', 'parallel_fwd',  'parallel_chunk_fwd',
                    'fused_chunk_fwdbwd', 'torch_fwdbwd', 'parallel_fwdbwd',
                    'parallel_chunk_fwdbwd'] + (['flash_fwd', 'flash_fwdbwd'] if HAS_FLASH else []),

        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('green', 'dotted'), ('blue', 'dotted'),
                ('red', 'dotted'), ('red', '--'), ('red', ':')] + ([('cyan', '-'), ('cyan', 'dotted')] if HAS_FLASH else []),
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
    B, H, D = 8, 16, 128

    if provider == 'flash' or provider == 'flash_bwd':
        q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    else:
        q = torch.randn(B, H, T, 16, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, H, T, 16, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    do = torch.ones_like(v, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'torch':
        if T > 1024:
            return results
        results = triton.testing.do_bench(lambda: naive_parallel_based(q, k, v), quantiles=quantiles)
    elif provider == 'fused_chunk':
        results = triton.testing.do_bench(lambda: fused_chunk_based(q, k, v), quantiles=quantiles)
    elif provider == 'parallel':
        results = triton.testing.do_bench(lambda: parallel_based(q, k, v), quantiles=quantiles)
    elif provider == 'parallel_chunk':
        results = triton.testing.do_bench(lambda: naive_chunk_based(q, k, v), quantiles=quantiles)
    elif provider == 'torch_bwd':
        if T > 1024:
            return results
        results = triton.testing.do_bench(lambda: naive_parallel_based(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'fused_chunk_bwd':
        results = triton.testing.do_bench(lambda: fused_chunk_based(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'parallel_bwd':
        results = triton.testing.do_bench(lambda: parallel_based(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'flash':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True), quantiles=quantiles)
    elif provider == 'flash_bwd':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True).backward(do), quantiles=quantiles)
    elif provider == 'parallel_chunk_bwd':
        results = triton.testing.do_bench(lambda: naive_chunk_based(q, k, v).backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True)
