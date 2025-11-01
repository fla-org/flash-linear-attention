
import os

import torch
import triton
from flash_attn import flash_attn_func
from torch.nn import functional as F

from fla.ops.comba import chunk_comba
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule
from fla.ops.kda import chunk_kda


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['gdn', 'comba', 'kda', 'dplr', 'attn'],
        # label name for the lines
        line_names=['gdn', 'comba', 'kda', 'dplr', 'attn'],
        # line styles
        styles=[('blue', '-'), ('red', '-.'), ('green', '-'), ('orange', '-.'),
                ('purple', '-'), ('brown', '-.'), ('pink', '-'), ('gray', '-.')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    ),
)
def benchmark(T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    B, H, D = 1, 16, 128

    # Set TMA environment variable based on provider
    original_tma_env = os.environ.get('FLA_USE_TMA', '0')

    if provider.endswith('_no_tma'):
        os.environ['FLA_USE_TMA'] = '0'
        provider_base = provider.replace('_no_tma', '')
    else:
        os.environ['FLA_USE_TMA'] = '1'
        provider_base = provider

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    do = torch.randn(B, T, H, D, dtype=dtype, device=device)
    if provider_base == 'gdn':
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = F.logsigmoid(torch.randn(B, T, H, dtype=dtype, device=device)).requires_grad_(True)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
        results = triton.testing.do_bench(
            lambda: chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )[0].backward(do),
            quantiles=quantiles,
        )
    elif provider_base == 'attn':
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        results = triton.testing.do_bench(
            lambda: flash_attn_func(
                q=q,
                k=k,
                v=v,
            ).backward(do),
            quantiles=quantiles,
        )
    elif provider_base == 'comba':
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        p = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float, device=device)).requires_grad_(True)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
        results = triton.testing.do_bench(
            lambda: chunk_comba(
                q=q,
                k=k,
                p=p,
                v=v,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )[0].backward(do),
            quantiles=quantiles,
        )
    elif provider_base == 'kda':
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)).requires_grad_(True)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
        results = triton.testing.do_bench(
            lambda: chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )[0].backward(do),
            quantiles=quantiles,
        )
    elif provider_base == 'dplr':
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        a = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        b = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)).requires_grad_(True)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
        results = triton.testing.do_bench(
            lambda: chunk_dplr_delta_rule(
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                gk=g,
            )[0].backward(do),
            quantiles=quantiles,
        )

    # Restore original TMA environment variable
    os.environ['FLA_USE_TMA'] = original_tma_env
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
