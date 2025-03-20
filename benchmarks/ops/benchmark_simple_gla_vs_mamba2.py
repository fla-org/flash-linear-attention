"""
Dependencies:
$ pip install mamba-ssm==2.2.2 triton==2.3.1

For correctness check, see:
https://github.com/sustcsonglin/flash-linear-attention/pull/49
"""

import torch
import triton
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from fla.ops.simple_gla import chunk_simple_gla


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[64] + [128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=["chunk_simple_gla", "mamba2_ssd"],
        # label name for the lines
        line_names=["chunk_simple_gla", "mamba2_ssd"],
        # line styles
        styles=[('blue', '-'), ('red', '-')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    # TODO: also add bwd pass benchmark
    from fla.utils import device
    dtype = torch.bfloat16
    B, H, D = 16, 8, 128
    # TODO: test more shapes
    # TODO: different values for D_V and D_QK
    # TODO: different values for H_Q and H_KV
    final_state = False  # does not impact performance

    # initialize Mamba2-format inputs
    X_mamba = 0.1 * torch.randn(B, T, H, D, dtype=dtype, device=device)
    dt_mamba = torch.ones(B, T, H, dtype=dtype, device=device)
    A_mamba = -0.1 * torch.rand(H, dtype=dtype, device=device)
    B_mamba = 0.1 * torch.randn(B, T, H, D, dtype=dtype, device=device)
    C_mamba = 0.1 * torch.randn(B, T, H, D, dtype=dtype, device=device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'chunk_simple_gla':
        # mapping inputs Mamba2 -> FLA
        # C, B, X: [B, T, H, D] -> [B, H, T, D]
        # g: [B, T, H] -> [B, H, T]
        q = C_mamba.transpose(1, 2).contiguous()
        k = B_mamba.transpose(1, 2).contiguous()
        v = X_mamba.transpose(1, 2).contiguous()
        g = (A_mamba * dt_mamba).transpose(1, 2).contiguous()
        # NOTE: whether to include the memory-copy cost of `contiguous()`?
        # this depends on the memory layout used by surrounding non-SSM layers

        results = triton.testing.do_bench(
            lambda: chunk_simple_gla(
                q, k, v, g, scale=1.0, output_final_state=final_state
            ), quantiles=quantiles
        )

    elif provider == 'mamba2_ssd':
        # NOTE: `chunk_size` is configurable in mamba2 kernel
        # here sets to the same hard-coded `BT = 64` as in simple_gla kernel
        # TODO: benchmark different chunk sizes
        results = triton.testing.do_bench(
            lambda:  mamba_chunk_scan_combined(
                X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
                chunk_size=64, D=None, return_final_states=final_state
            ),
            quantiles=quantiles
        )
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, save_path='.')
