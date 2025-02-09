# -*- coding: utf-8 -*-

import torch
import triton

from fla.ops.based import parallel_based
from fla.ops.gla import fused_chunk_gla
from fla.ops.retention import fused_chunk_retention, parallel_retention

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['retention_parallel', 'retention_fused_chunk',
                   'gla_fused_chunk', 'based_parallel'] + (['flash'] if HAS_FLASH else []),
        # label name for the lines
        line_names=['retention_parallel_fwdbwd', 'retention_fused_chunk_fwdbwd',
                    'gla_fused_chunk_fwdbwd', 'based_parallel_fwdbwd'] + (['flash_fwdbwd'] if HAS_FLASH else []),
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')] + \
        ([('yellow', 'dotted')] if HAS_FLASH else []),
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

    if provider == 'flash':
        q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    elif "based" in provider:
        q = torch.randn(B, H, T, 16, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, H, T, 16, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    elif "gla" in provider:
        q = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
        g = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    else:
        q = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)

    do = torch.rand_like(v, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'flash':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'retention_parallel':
        results = triton.testing.do_bench(lambda: parallel_retention(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'retention_fused_chunk':
        results = triton.testing.do_bench(lambda: fused_chunk_retention(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'based_parallel':
        results = triton.testing.do_bench(lambda: parallel_based(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'gla_fused_chunk':
        results = triton.testing.do_bench(lambda: fused_chunk_gla(q, k, v, g).backward(do), quantiles=quantiles)

    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True)
