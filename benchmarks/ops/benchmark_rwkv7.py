# -*- coding: utf-8 -*-


import torch
import triton

from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7, torch_addcmul_rwkv7


def addcmul_torch_fuse(hidden_states, delta, x_x):
    return hidden_states.addcmul(delta, x_x.view(6, 1, 1, -1)).unbind(0)


@torch.jit.script
def torch_jit_addcmul(hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g):
    xr = hidden_states + delta * x_r
    xw = hidden_states + delta * x_w
    xk = hidden_states + delta * x_k
    xv = hidden_states + delta * x_v
    xa = hidden_states + delta * x_a
    xg = hidden_states + delta * x_g
    return xr, xw, xk, xv, xa, xg


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        line_vals=['addcmul_torch', 'addcmul_torch_fuse', 'addcmul_triton', 'torch_jit',
                   'addcmul_torch_bwd', 'addcmul_torch_fuse_bwd', 'addcmul_triton_bwd', 'torch_jit_bwd'],
        # label name for the lines
        line_names=['addcmul_torch', 'addcmul_torch_fuse', 'addcmul_triton', 'torch_jit',
                    'addcmul_torch_bwd', 'addcmul_torch_fuse_bwd', 'addcmul_triton_bwd', 'torch_jit_bwd'],

        # line styles
        styles=[
            ('green', '-'),
            ('blue', '--'),
            ('red', '-.'),
            ('cyan', ':'),
            ('magenta', '-'),
            ('yellow', 'dotted'),
            ('black', 'dashdot'),
            ('orange', 'solid'),

        ],
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
    # Read B, H, D from environment variables, default to 16, 8, 128 if not set
    hidden_size = 4096
    batch_size = 8
    seq_len = T
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device,
                                requires_grad=requires_grad, dtype=dtype).to(device)
    delta = torch.randn_like(hidden_states).to(device)
    x_x_fuse = torch.randn(6, hidden_size).to(device)
    x_r = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    x_w = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    x_k = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    x_v = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    x_a = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    x_g = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'addcmul_torch_fuse':
        results = triton.testing.do_bench(lambda: addcmul_torch_fuse(hidden_states, delta, x_x_fuse), quantiles=quantiles)
    elif provider == 'addcmul_torch':
        results = triton.testing.do_bench(lambda: torch_addcmul_rwkv7(
            hidden_states, delta,  x_r, x_w, x_k, x_v, x_a, x_g), quantiles=quantiles)
    elif provider == 'torch_jit':
        results = triton.testing.do_bench(lambda: torch_jit_addcmul(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g), quantiles=quantiles)
    elif provider == 'addcmul_triton':
        results = triton.testing.do_bench(lambda: fused_addcmul_rwkv7(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g), quantiles=quantiles)
    elif provider == 'addcmul_torch_bwd':
        results = triton.testing.do_bench(lambda: (lambda outputs: sum([o.sum() for o in outputs]).backward())(
            torch_addcmul_rwkv7(hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g)))
    elif provider == 'addcmul_torch_fuse_bwd':
        results = triton.testing.do_bench(lambda: (lambda outputs: sum([o.sum() for o in outputs]).backward())(
            addcmul_torch_fuse(hidden_states, delta, x_x_fuse)))
    elif provider == 'addcmul_triton_bwd':
        results = triton.testing.do_bench(lambda: (lambda outputs: sum([o.sum() for o in outputs]).backward())(
            fused_addcmul_rwkv7(hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g)))
    elif provider == 'torch_jit_bwd':
        results = triton.testing.do_bench(lambda: (lambda outputs: sum([o.sum() for o in outputs]).backward())(
            torch_jit_addcmul(hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g)))

    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
