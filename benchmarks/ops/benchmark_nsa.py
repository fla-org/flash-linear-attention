# -*- coding: utf-8 -*-

import torch
import triton
from flash_attn import flash_attn_func

from fla.ops.nsa import parallel_nsa


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['nsa', 'nsa_bwd', 'flash', 'flash_bwd'],
        # label name for the lines
        line_names=['nsa', 'nsa_bwd', 'flash', 'flash_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('green', 'dotted'),
                ('blue', 'dotted'), ('red', 'dotted'), ('cyan', '-'), ('cyan', 'dotted')],
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
    B, H, HQ, D, S = 4, 4, 64, 128, 16
    block_size = 64

    q = torch.randn(B, T, HQ, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    do = torch.ones_like(q, dtype=dtype)

    indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                indices[b, t, h, :len(i_i)] = i_i
    indices = indices.sort(-1)[0]

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'nsa':
        results = triton.testing.do_bench(
            lambda: parallel_nsa(q, k, v, indices, block_size),
            quantiles=quantiles
        )
    elif provider == 'nsa_bwd':
        results = triton.testing.do_bench(
            lambda: parallel_nsa(q, k, v, indices, block_size).backward(do),
            quantiles=quantiles
        )
    elif provider == 'flash':
        results = triton.testing.do_bench(
            lambda: flash_attn_func(q, k, v, causal=True),
            quantiles=quantiles
        )
    elif provider == 'flash_bwd':
        results = triton.testing.do_bench(
            lambda: flash_attn_func(q, k, v, causal=True).backward(do),
            quantiles=quantiles
        )
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, save_path='.')
