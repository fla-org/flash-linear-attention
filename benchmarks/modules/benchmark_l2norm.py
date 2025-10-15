
from functools import partial

import torch
import torch.nn.functional as F
import triton

from fla.modules.l2norm import l2norm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['B', 'T', 'H', 'D'],
        # different possible values for `x_name`
        x_vals=[(16, 128 * 2 ** i, h, 2048//h) for h in [1, 2, 4, 8, 16] for i in range(1, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['naive', 'compiled',  'fused', 'naive_bwd', 'compiled_bwd',  'fused_bwd'],
        # label name for the lines
        line_names=['naive', 'compiled',  'fused', 'naive_bwd', 'compiled_bwd',  'fused_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'),
                ('cyan', ':'), ('yellow', 'dotted'), ('cyan', '--'), ('cyan', '-'), ('black', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    ),
)
def benchmark(B, H, D, T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    requires_grad = True
    x = torch.randn(B * T, D, device=device, requires_grad=requires_grad, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider.startswith('naive'):
        norm = partial(F.normalize, dim=-1, p=2)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('compiled'):
        norm = torch.compile(partial(F.normalize, dim=-1, p=2))
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('fused'):
        norm = l2norm
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('naive_bwd'):
        norm = partial(F.normalize, dim=-1, p=2)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    if provider.startswith('compiled_bwd'):
        norm = torch.compile(partial(F.normalize, dim=-1, p=2))
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    if provider.startswith('fused_bwd'):
        norm = l2norm
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
