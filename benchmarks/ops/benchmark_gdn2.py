# -*- coding: utf-8 -*-

import os

import torch
import triton
from torch.nn import functional as F

from fla.ops.gdn2 import chunk_gdn2
from fla.ops.gdn2.gate import fused_gdn2_gate, gdn2_gate_ref


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[4096, 8192, 16384],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['chunk_gdn2_no_tma', 'chunk_gdn2',  'chunk_gdn2_bwd_no_tma', 'chunk_gdn2_bwd',
                   'gdn2_gate_fused', 'gdn2_gate_ref', 'gdn2_gate_fused_bwd', 'gdn2_gate_ref_bwd'],
        # label name for the lines
        line_names=['chunk_gdn2_no_tma', 'chunk_gdn2', 'chunk_gdn2_fwdbwd_no_tma', 'chunk_gdn2_fwdbwd',
                    'gdn2_gate_fused', 'gdn2_gate_ref', 'gdn2_gate_fused_bwd', 'gdn2_gate_ref_bwd'],
        # line styles
        styles=[('blue', '-'), ('red', '-.'), ('green', '-'), ('orange', '-.'),
                ('purple', '-'), ('brown', '-.'), ('pink', '-'), ('gray', '-.')],
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
        os.environ['FLA_USE_TMA'] = '0'
        provider_base = provider.replace('_no_tma', '')
    else:
        os.environ['FLA_USE_TMA'] = '1'
        provider_base = provider

    with torch.no_grad():
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = (F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float, device=device))
             / gate_logit_normalizer).requires_grad_(True)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
        h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device).requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    # GDN2 gate benchmark setup
    if provider_base in ['gdn2_gate_fused', 'gdn2_gate_ref', 'gdn2_gate_fused_bwd', 'gdn2_gate_ref_bwd']:
        # Setup for gate benchmark - different tensor shapes
        B_gate, H_gate, D_gate = 16, 8, 128  # Same as main benchmark
        g_input = torch.randn(B_gate, T, H_gate * D_gate, dtype=dtype, device=device).requires_grad_(True)
        A_param = torch.randn(H_gate, dtype=dtype, device=device).requires_grad_(True)

        if provider_base == 'gdn2_gate_fused':
            results = triton.testing.do_bench(
                lambda: fused_gdn2_gate(g_input, A_param, D_gate),
                quantiles=quantiles
            )
        elif provider_base == 'gdn2_gate_ref':
            results = triton.testing.do_bench(
                lambda: gdn2_gate_ref(g_input, A_param, D_gate),
                quantiles=quantiles
            )
        elif provider_base == 'gdn2_gate_fused_bwd':
            do_gate = torch.randn(B_gate, T, H_gate, D_gate, dtype=dtype, device=device)

            def fused_bench():
                g_out = fused_gdn2_gate(g_input, A_param, D_gate)
                g_out.backward(do_gate, retain_graph=True)
                return g_out
            results = triton.testing.do_bench(fused_bench, quantiles=quantiles)
        elif provider_base == 'gdn2_gate_ref_bwd':
            do_gate = torch.randn(B_gate, T, H_gate, D_gate, dtype=dtype, device=device)

            def ref_bench():
                g_out = gdn2_gate_ref(g_input, A_param, D_gate)
                g_out.backward(do_gate, retain_graph=True)
                return g_out
            results = triton.testing.do_bench(ref_bench, quantiles=quantiles)
    else:
        # Original chunk_gdn2 benchmarks
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
                    use_qk_l2norm_in_kernel=True
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
                    use_qk_l2norm_in_kernel=True
                )[0].backward(do),
                quantiles=quantiles
            )

    # Restore original TMA environment variable
    os.environ['FLA_NO_USE_TMA'] = original_tma_env
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
