# -*- coding: utf-8 -*-

import torch
import triton
from torch.nn import functional as F

from fla.ops.gla import chunk_gla
from fla.ops.retention import chunk_retention
from fla.ops.rwkv6 import chunk_rwkv6

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except BaseException:
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
        line_vals=['rwkv6',  'gla', 'rwkv6_bwd', 'gla_bwd', 'retention_bwd', 'flash_bwd'],
        # label name for the lines
        line_names=['rwkv6',  'gla', 'rwkv6_bwd', 'gla_bwd', 'retention_bwd', 'flash_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'),
                ('cyan', ':'), ('yellow', 'dotted'), ('black', ':')],
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
    v = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    if provider.startswith('flash'):
        q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    if provider.startswith('gla'):
        g = F.logsigmoid(torch.randn(B, H, T, D, device=device, dtype=dtype))
        g = g.clamp_min(-5).requires_grad_(requires_grad)
    if provider.startswith('rwkv6'):
        w = F.logsigmoid(torch.randn(B, H, T, D, device=device, dtype=dtype)).requires_grad_(True)
        u = torch.randn(H, D, device=device, dtype=dtype).requires_grad_(True)

    do = torch.ones_like(v, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'rwkv6':
        results = triton.testing.do_bench(lambda: chunk_rwkv6(q, k, v, w, u), quantiles=quantiles)
    elif provider == 'gla':
        results = triton.testing.do_bench(lambda: chunk_gla(q, k, v, g), quantiles=quantiles)
    elif provider == 'rwkv6_bwd':
        results = triton.testing.do_bench(lambda: chunk_rwkv6(q, k, v, w, u)[0].backward(do), quantiles=quantiles)
    elif provider == 'gla_bwd':
        results = triton.testing.do_bench(lambda: chunk_gla(q, k, v, g)[0].backward(do), quantiles=quantiles)
    elif provider == 'retention_bwd':
        results = triton.testing.do_bench(lambda: chunk_retention(q, k, v)[0].backward(do), quantiles=quantiles)
    elif provider == 'flash_bwd':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True).backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
