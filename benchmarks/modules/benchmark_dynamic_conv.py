# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton

from fla.modules.dynamic_conv import dynamic_conv1d
from fla.modules.dynamic_conv.naive import naive_dynamic_conv1d
from fla.utils import device


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['B', 'T', 'D', 'R', 'packed', 'activation'],
        x_vals=[(*shape, activation) for activation in ['identity', 'swish'] for shape in
                [(2, 256, 512, 16, False), (4, 4096, 2048, 16, False),
                 (1, 8192, 1024, 32, True), (16, 1, 2048, 16, False)]],
        line_arg='provider',
        line_vals=['torch-fwd', 'triton-fwd', 'torch-fwdbwd', 'triton-fwdbwd'],
        line_names=['PyTorch forward', 'Triton forward', 'PyTorch forward+backward', 'Triton forward+backward'],
        styles=[('blue', '--'), ('green', '--'), ('blue', '-'), ('green', '-')],
        ylabel='Latency (ms)',
        plot_name='dynamic-conv',
        args={},
    ),
)
def benchmark(B, T, D, R, packed, activation, provider):
    torch.manual_seed(42)
    W = 4
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16).requires_grad_()
    z = (torch.randn(B, T, R, device=device, dtype=torch.bfloat16) / R**0.5).requires_grad_()
    weight = torch.randn(R, W, D, device=device, dtype=torch.bfloat16).requires_grad_()
    kernel_bias = torch.randn(W, D, device=device, dtype=torch.bfloat16).requires_grad_()
    cu = torch.tensor([0, 1, 2047, T], device=device, dtype=torch.int32) if packed else None
    # the reference reads packed metadata on the host; exclude that synchronization from timing.
    ref_cu = cu.cpu() if cu is not None else None
    state = torch.randn(B, D, W, device=device, dtype=x.dtype) if T == 1 else None
    fn = dynamic_conv1d if provider.startswith('triton') else naive_dynamic_conv1d
    dy = torch.randn_like(x)

    def run():
        y, _ = fn(
            x=x,
            z=z,
            weight=weight,
            kernel_bias=kernel_bias,
            residual=x,
            initial_state=state,
            output_final_state=T == 1,
            activation=None if activation == 'identity' else activation,
            cu_seqlens=cu if provider.startswith('triton') else ref_cu,
        )
        if provider.endswith('fwdbwd'):
            torch.autograd.grad(y, (x, z, weight, kernel_bias), dy)

    return triton.testing.do_bench(run, quantiles=[0.5, 0.2, 0.8])


if __name__ == '__main__':
    from benchmarks.modules.runner import run_module_benchmark

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_module_benchmark(benchmark, script_file=__file__, visualize=False)
