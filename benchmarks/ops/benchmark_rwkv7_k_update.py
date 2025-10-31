import torch
import triton

from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7


@torch.jit.script
def k_update_ref(k: torch.Tensor, a: torch.Tensor, ka: torch.Tensor) -> torch.Tensor:
    return k.addcmul(k * (a - 1), ka)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 9)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['naive_k_update',  'fused_k_update', 'naive_k_update_bwd',  'fused_k_update_bwd'],
        # label name for the lines
        line_names=['naive_k_update',  'fused_k_update', 'naive_k_update_bwd',  'fused_k_update_bwd'],
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
    a = torch.randn(B, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    ka = torch.randn(1, 1, D, device=device, requires_grad=requires_grad, dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider.startswith('naive_k_update'):
        results = triton.testing.do_bench(lambda: k_update_ref(x, a, ka), quantiles=quantiles)
    if provider.startswith('fused_k_update'):
        results = triton.testing.do_bench(lambda: fused_k_rwkv7(x, a, ka), quantiles=quantiles)
    if provider.startswith('naive_k_update_bwd'):
        grad_output = torch.randn_like(x)
        results = triton.testing.do_bench(lambda: k_update_ref(x, a, ka).backward(grad_output), quantiles=quantiles)
    if provider.startswith('fused_k_update_bwd'):
        grad_output = torch.randn_like(x)
        results = triton.testing.do_bench(lambda: fused_k_rwkv7(x, a, ka).backward(grad_output), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
