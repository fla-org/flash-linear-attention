# -*- coding: utf-8 -*-

import torch
import triton
from torch.nn import functional as F

from fla.ops.gla import chunk_gla
from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa
from fla.ops.retention import chunk_retention

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
        line_vals=['gsa_recurrent', 'gsa_chunk', 'gla',
                   'gsa_recurrent_bwd', 'gsa_chunk_bwd', 'gla_bwd', 'retention_bwd', 'flash_bwd'],
        # label name for the lines
        line_names=['gsa_recurrent', 'gsa_chunk', 'gla',
                    'gsa_recurrent_bwd', 'gsa_chunk_bwd', 'gla_bwd', 'retention_bwd', 'flash_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'),
                ('cyan', ':'), ('yellow', 'dotted'), ('black', ':'), ('green', ':'), ('green', 'dotted'), ('green', ':')],
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
    B, H, D, M = 16, 4, 128, 64

    q = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    if provider.startswith('gsa'):
        f = F.logsigmoid(torch.randn(B, H, T, M, device=device, dtype=dtype))
        s = (1 - f.exp()).to(f.dtype)
    if provider.startswith('gla'):
        g = F.logsigmoid(torch.randn(B, H, T, D, device=device, dtype=dtype))
        g = g.clamp_min(-5).requires_grad_(requires_grad)
    if provider.startswith('flash'):
        q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    do = torch.ones_like(v, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'gsa_recurrent':
        return triton.testing.do_bench(lambda: fused_recurrent_gsa(q, k, v, s, f), quantiles=quantiles)
    if provider == 'gsa_chunk':
        return triton.testing.do_bench(lambda: chunk_gsa(q, k, v, s, f), quantiles=quantiles)
    elif provider == 'gla':
        return triton.testing.do_bench(lambda: chunk_gla(q, k, v, g), quantiles=quantiles)
    elif provider == 'gsa_recurrent_bwd':
        return triton.testing.do_bench(lambda: fused_recurrent_gsa(q, k, v, s, f)[0].backward(do), quantiles=quantiles)
    elif provider == 'gsa_chunk_bwd':
        return triton.testing.do_bench(lambda: chunk_gsa(q, k, v, s, f)[0].backward(do), quantiles=quantiles)
    elif provider == 'gla_bwd':
        return triton.testing.do_bench(lambda: chunk_gla(q, k, v, g)[0].backward(do), quantiles=quantiles)
    elif provider == 'retention_bwd':
        return triton.testing.do_bench(lambda: chunk_retention(q, k, v)[0].backward(do), quantiles=quantiles)
    elif provider == 'flash_bwd':
        return triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True).backward(do), quantiles=quantiles)


if __name__ == '__main__':
    benchmark.run(print_data=True)
