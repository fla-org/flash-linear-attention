# -*- coding: utf-8 -*-

import torch
import triton

from fla.ops.hgrn import chunk_hgrn, fused_recurrent_hgrn


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['chunk', 'recurrent', 'chunk_bwd', 'recurrent_bwd'],
        # label name for the lines
        line_names=['chunk', 'recurrent', 'chunk_bwd', 'recurrent_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':'), ('yellow', 'dotted'), ('black', 'dashed')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    B, D = 16, 512

    x = torch.randn((B, T, D), dtype=dtype, device=device)
    g = torch.randn((B, T, D), dtype=dtype, device=device).sigmoid()
    x = (1 - g) * x
    x, g = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g))
    do = torch.randn_like(x, dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'chunk':
        results = triton.testing.do_bench(lambda: chunk_hgrn(x, g), quantiles=quantiles)
    if provider == 'recurrent':
        results = triton.testing.do_bench(lambda: fused_recurrent_hgrn(x, g), quantiles=quantiles)
    if provider == 'chunk_bwd':
        results = triton.testing.do_bench(lambda: chunk_hgrn(x, g)[0].backward(do), quantiles=quantiles)
    if provider == 'recurrent_bwd':
        results = triton.testing.do_bench(lambda: fused_recurrent_hgrn(x, g)[0].backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
