# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
from einops import rearrange

from fla.modules.convolution import causal_conv1d
from fla.ops.utils.index import prepare_sequence_ids
from fla.utils import IS_NPU

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def _conv_benchmark_providers():
    providers = ['causal_conv1d_fwd', 'causal_conv1d_fwdbwd']
    if not IS_NPU and causal_conv1d_fn is not None:
        providers.extend(['causal_conv1d_cuda_fwd', 'causal_conv1d_cuda_fwdbwd'])
    return providers


_LINE_VALS = _conv_benchmark_providers()
_STYLES = [
    ('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':'),
    ('yellow', 'dotted'), ('cyan', '--'), ('cyan', '-'), ('black', ':'),
]


def benchmark(T, D, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    requires_grad = True
    B, N, W = 1, 16, 4
    if T < 2048:
        N = 4

    x = torch.randn(B, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    weight = torch.randn(D, W).to(device)
    bias = torch.randn(D).to(device)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long),
    ], 0).to(device).sort()[0]
    if provider.startswith('causal_conv1d_fwdbwd'):
        results = triton.testing.do_bench(
            lambda: causal_conv1d(x, weight, bias, activation='swish', cu_seqlens=cu_seqlens)[0].backward(x),
            quantiles=quantiles,
        )
    elif provider.startswith('causal_conv1d_cuda_fwdbwd'):
        results = triton.testing.do_bench(
            lambda: rearrange(
                causal_conv1d_fn(
                    x=rearrange(x, 'b t d -> b d t'),
                    weight=weight,
                    bias=bias,
                    activation='swish',
                    seq_idx=prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0),
                ),
                'b d t -> b t d',
            ).backward(x),
            quantiles=quantiles,
        )
    elif provider.startswith('causal_conv1d_fwd'):
        results = triton.testing.do_bench(
            lambda: causal_conv1d(x, weight, bias, activation='swish', cu_seqlens=cu_seqlens),
            quantiles=quantiles,
        )
    elif provider.startswith('causal_conv1d_cuda_fwd'):
        results = triton.testing.do_bench(
            lambda: rearrange(
                causal_conv1d_fn(
                    x=rearrange(x, 'b t d -> b d t'),
                    weight=weight,
                    bias=bias,
                    activation='swish',
                    seq_idx=prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0),
                ),
                'b d t -> b t d',
            ),
            quantiles=quantiles,
        )
    return results


benchmark = triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T', 'D'],
        # different possible values for `x_name`
        x_vals=[(128 * 2 ** i, d) for d in [256, 512, 1024, 2048, 4096] for i in range(1, 10)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=_LINE_VALS,
        # label name for the lines
        line_names=_LINE_VALS,
        # line styles
        styles=_STYLES[:len(_LINE_VALS)],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    ),
)(benchmark)


if __name__ == '__main__':
    try:
        from runner import run_module_benchmark
    except ModuleNotFoundError:
        from benchmarks.modules.runner import run_module_benchmark

    run_module_benchmark(benchmark, script_file=__file__)
