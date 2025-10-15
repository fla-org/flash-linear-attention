import torch
import torch.nn as nn
import triton

from fla.modules.token_shift import token_shift


def token_shift_ref(x):
    shifted = nn.functional.pad(x, (0, 0, 1, -1))
    delta = shifted - x
    return delta


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 9)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['naive_token_shift',  'fused_token_shift', 'naive_token_shift_bwd',  'fused_token_shift_bwd'],
        # label name for the lines
        line_names=['naive_token_shift',  'fused_token_shift', 'naive_token_shift_bwd',  'fused_token_shift_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'),
                ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    ),
)
def benchmark(T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    requires_grad = True
    B, D = 8, 4096

    x = torch.randn(B, T, D, device=device, requires_grad=requires_grad, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider.startswith('naive_token_shift'):
        results = triton.testing.do_bench(lambda: token_shift_ref(x), quantiles=quantiles)
    if provider.startswith('fused_token_shift'):
        results = triton.testing.do_bench(lambda: token_shift(x), quantiles=quantiles)
    if provider.startswith('naive_token_shift_bwd'):
        grad_output = torch.randn_like(x)
        results = triton.testing.do_bench(lambda: token_shift_ref(x).backward(grad_output), quantiles=quantiles)
    if provider.startswith('fused_token_shift_bwd'):
        grad_output = torch.randn_like(x)
        results = triton.testing.do_bench(lambda: token_shift(x).backward(grad_output), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
