# -*- coding: utf-8 -*-

import os

import torch
import triton
from torch.nn import functional as F

from fla.ops.gdn2 import chunk_gdn2


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[4096, 8192, 16384],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['chunk_gdn2_no_tma', 'chunk_gdn2',  'chunk_gdn2_bwd_no_tma', 'chunk_gdn2_bwd'],
        # label name for the lines
        line_names=['chunk_gdn2_no_tma', 'chunk_gdn2', 'chunk_gdn2_fwdbwd_no_tma', 'chunk_gdn2_fwdbwd'],
        # line styles
        styles=[('blue', '-'), ('red', '-.'), ('green', '-'), ('orange', '-.')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    B, H, D = 16, 8, 128
    scale = 1.0
    gate_logit_normalizer = 1.0

    # Set TMA environment variable based on provider
    original_tma_env = os.environ.get('FLA_NO_USE_TMA', '0')

    if provider.endswith('_no_tma'):
        os.environ['FLA_NO_USE_TMA'] = '1'
        provider_base = provider.replace('_no_tma', '')
    else:
        os.environ['FLA_NO_USE_TMA'] = '0'
        provider_base = provider

    with torch.no_grad():
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = (F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float, device=device)) / gate_logit_normalizer).requires_grad_(True)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
        h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device).requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    if provider_base == 'chunk_gdn2':
        results = triton.testing.do_bench(
            lambda: chunk_gdn2(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                use_q_l2norm=True,
                use_k_l2norm=True
            ),
            quantiles=quantiles
        )
    elif provider_base == 'chunk_gdn2_bwd':
        do = torch.ones_like(v, dtype=dtype)
        results = triton.testing.do_bench(
            lambda: chunk_gdn2(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                use_q_l2norm=True,
                use_k_l2norm=True
            )[0].backward(do),
            quantiles=quantiles
        )

    # Restore original TMA environment variable
    os.environ['FLA_NO_USE_TMA'] = original_tma_env
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
