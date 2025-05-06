# -*- coding: utf-8 -*-

import torch
import triton
from einops import rearrange

from fla.modules.canon import canon
from fla.ops.utils.index import prepare_sequence_ids

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T', 'D'],
        # different possible values for `x_name`
        x_vals=[(128 * 2 ** i, d) for d in [256, 512, 1024, 2048, 4096] for i in range(1, 10)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['canon', 'causal_conv1d'],
        # label name for the lines
        line_names=['canon', 'causal_conv1d'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'),
                ('cyan', ':'), ('yellow', 'dotted'), ('cyan', '--'), ('cyan', '-'), ('black', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, D, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    requires_grad = True
    B, N, W = 1, 16, 4
    if T < 2048:
        N = 4

    x = torch.randn(B, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    weight = torch.randn(D, W).to(device)
    bias = torch.randn(D).to(device)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    if provider.startswith('canon'):
        results = triton.testing.do_bench(
            lambda: canon(x, weight, bias, activation='swish', cu_seqlens=cu_seqlens),
            quantiles=quantiles
        )
    if provider.startswith('causal_conv1d'):
        results = triton.testing.do_bench(
            lambda: rearrange(
                causal_conv1d_fn(
                    x=rearrange(x, 'b t d -> b d t'),
                    weight=weight,
                    bias=bias,
                    activation='swish',
                    seq_idx=prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0),
                ),
                'b d t -> b t d'
            ),
            quantiles=quantiles
        )
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
