# -*- coding: utf-8 -*-

import os

import torch
import triton
from torch.nn import functional as F

from fla.ops.gla import chunk_gla
from fla.ops.retention import chunk_retention
from fla.ops.rwkv6 import chunk_rwkv6
from fla.ops.rwkv7 import chunk_rwkv7

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
        # line styles
        line_vals=['rwkv6', 'rwkv7', 'gla', 'flash', 'rwkv6_bwd', 'rwkv7_bwd', 'gla_bwd', 'retention_bwd', 'flash_bwd'],
        # label name for the lines
        line_names=['rwkv6', 'rwkv7', 'gla', 'flash', 'rwkv6_bwd', 'rwkv7_bwd', 'gla_bwd', 'retention_bwd', 'flash_bwd'],
        # # line styles
        styles=[
            ('green', '-'),      # rwkv6
            ('blue', '--'),      # rwkv7
            ('red', '-.'),       # gla
            ('cyan', ':'),       # rwkv6_bwd
            ('magenta', '-'),    # rwkv7_bwd
            ('yellow', 'dotted'),  # gla_bwd
            ('black', ':'),      # retention_bwd
            ('gray', ':'),       # flash
            ('gray', '--')       # flash_bwd
        ],
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
    # Read B, H, D from environment variables, default to 16, 8, 128 if not set
    B = int(os.getenv('BENCH_B', '8'))  # Batch size
    H = int(os.getenv('BENCH_H', '64'))   # Number of heads
    D = int(os.getenv('BENCH_D', '64'))  # Dimension per head
    with torch.no_grad():
        q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        if provider.startswith('flash'):
            q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
            k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
            v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        if provider.startswith('gla'):
            g = F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=dtype))
            g = g.clamp_min(-5).requires_grad_(requires_grad)
        if provider.startswith('rwkv6'):
            w = F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=dtype)).requires_grad_(True)
            u = torch.randn(H, D, device=device, dtype=dtype).requires_grad_(True)
        if provider.startswith('rwkv7'):
            q = torch.empty(B, T, H, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
            k = torch.empty(B, T, H, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
            v = torch.empty(B, T, H, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
            w = F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=dtype)).requires_grad_(True)
            kk = torch.empty(B, T, H, D, device=device).uniform_(-1, 1)
            kk = torch.nn.functional.normalize(kk, dim=-1).to(dtype=dtype)

            a = -kk.clone().requires_grad_(True)  # -kk
            a_scale = torch.empty(B, T, H, D, device=device).uniform_(0, 0.1).to(dtype=dtype)
            b = (kk * a_scale).requires_grad_(True)  # kk*a

    do = torch.ones_like(v, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'rwkv6':
        results = triton.testing.do_bench(lambda: chunk_rwkv6(q, k, v, w, u), quantiles=quantiles)
    elif provider == 'rwkv7':
        results = triton.testing.do_bench(lambda: chunk_rwkv7(q, w, k, v, a, b), quantiles=quantiles)
    elif provider == 'gla':
        results = triton.testing.do_bench(lambda: chunk_gla(q, k, v, g), quantiles=quantiles)
    elif provider == 'rwkv6_bwd':
        results = triton.testing.do_bench(lambda: chunk_rwkv6(q, k, v, w, u)[0].backward(do), quantiles=quantiles)
    elif provider == 'rwkv7_bwd':
        results = triton.testing.do_bench(lambda: chunk_rwkv7(q, w, k, v, a, b)[0].backward(do), quantiles=quantiles)
    elif provider == 'gla_bwd':
        results = triton.testing.do_bench(lambda: chunk_gla(q, k, v, g)[0].backward(do), quantiles=quantiles)
    elif provider == 'retention_bwd':
        results = triton.testing.do_bench(lambda: chunk_retention(q, k, v)[0].backward(do), quantiles=quantiles)
    elif provider == 'flash':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True), quantiles=quantiles)
    elif provider == 'flash_bwd':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True).backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
