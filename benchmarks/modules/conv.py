# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import os
import statistics
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
from einops import rearrange

from benchmarks.utils.distributed import distributed_context
from benchmarks.utils.perf_report import add_module_arguments, run_module_benchmark
from benchmarks.utils.reporting import BenchmarkResult
from fla.modules.convolution import causal_conv1d
from fla.ops.utils.index import prepare_sequence_ids
from fla.utils import IS_NPU, IS_NVIDIA, find_spec_cached

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed import ProcessGroup

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def _conv_benchmark_providers():
    providers = ['causal_conv1d_fwd', 'causal_conv1d_fwdbwd']
    if not IS_NPU and causal_conv1d_fn is not None:
        providers.extend(['causal_conv1d_cuda_fwd', 'causal_conv1d_cuda_fwdbwd'])
    if IS_NVIDIA and find_spec_cached('triton.experimental.gluon') is not None:
        providers.extend(['causal_conv1d_gluon_fwd', 'causal_conv1d_gluon_fwdbwd'])
    return providers


_LINE_VALS = _conv_benchmark_providers()
_STYLES = [
    ('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':'),
    ('yellow', 'dotted'), ('cyan', '--'), ('cyan', '-'), ('black', ':'),
]


def benchmark(T, D, provider, packed=True):
    from fla.utils import device
    torch.manual_seed(42)
    # select each provider independently of inherited Gluon switches.
    os.environ['FLA_GLUON'] = '0'
    os.environ['FLA_CONV_GLUON'] = '1' if '_gluon_' in provider else '0'
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
    ], 0).to(device).sort()[0] if packed else None
    if provider in ('causal_conv1d_fwdbwd', 'causal_conv1d_gluon_fwdbwd'):
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
                    seq_idx=prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0) if packed else None,
                ),
                'b d t -> b t d',
            ).backward(x),
            quantiles=quantiles,
        )
    elif provider in ('causal_conv1d_fwd', 'causal_conv1d_gluon_fwd'):
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
                    seq_idx=prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0) if packed else None,
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


def _compare_cp_backends(
    forward: Callable[[str], torch.Tensor],
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    group: ProcessGroup | None,
) -> None:
    from fla.utils import assert_close, get_abs_err, get_err_ratio

    results = []
    for backend in ('triton', 'cuda'):
        y = forward(backend)
        dx, dw = torch.autograd.grad(outputs=y, inputs=(x, weight), grad_outputs=dy)
        dw = dw.float()
        if group is not None:
            dist.all_reduce(tensor=dw, group=group)
        results.append((y.detach(), dx, dw))

    for name, ref, actual in zip(('y', 'dx', 'dw'), *results):
        ref, actual = ref.float(), actual.float()
        errors = torch.tensor([get_abs_err(ref, actual), get_err_ratio(ref, actual)], device=x.device)
        if group is not None:
            dist.all_reduce(tensor=errors, op=dist.ReduceOp.MAX, group=group)
        assert torch.isfinite(ref).all() and torch.isfinite(actual).all(), name
        assert_close(name, ref, actual, ratio=0.005)
        if group is None or dist.get_rank(group=group) == 0:
            print(
                f'  {name:6s} CUDA vs Triton: max_abs={errors[0].item():.3e}, relative_rms={errors[1].item():.3e}',
                flush=True,
            )


def _measure_cp(
    step: Callable[[str], None],
    repeats: int,
    iterations: int,
    group: ProcessGroup | None,
) -> dict[str, list[float]]:
    for backend in ('triton', 'cuda'):
        for _ in range(5):
            step(backend)
    torch.cuda.synchronize()

    samples = {'triton': [], 'cuda': []}
    for _ in range(repeats):
        for backend in samples:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if group is not None:
                dist.barrier(group=group)
            start.record()
            for _ in range(iterations):
                step(backend)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) / iterations
            if group is not None:
                elapsed_tensor = torch.tensor(elapsed, device='cuda')
                dist.all_reduce(tensor=elapsed_tensor, op=dist.ReduceOp.MAX, group=group)
                elapsed = elapsed_tensor.item()
            samples[backend].append(elapsed)
    return samples


@torch.compiler.disable
def _benchmark_cp_case(tokens: int, channels: int, packed: bool, repeats: int, iterations: int, cp: bool):
    from fla.modules.causal_conv1d import causal_conv1d as causal_conv1d_backend
    from fla.ops.cp import build_cp_context

    torch.manual_seed(42)
    group = dist.group.WORLD if cp else None
    global_tokens = tokens * dist.get_world_size(group=group) if cp else tokens
    lengths = [global_tokens]
    if packed:
        quarter = global_tokens // 4
        lengths = [1, 2, 61, quarter - 64, quarter + (7 if cp else 0)]
        lengths.append(global_tokens - sum(lengths))
    cu_seqlens_cpu = F.pad(torch.tensor(lengths).cumsum(dim=0, dtype=torch.int32), (1, 0))
    cu_seqlens = cu_seqlens_cpu.to(device='cuda')
    cp_context = None
    if cp:
        cp_context = build_cp_context(
            cu_seqlens=cu_seqlens,
            group=group,
            conv1d_kernel_size=4,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    x = torch.randn(1, tokens, channels, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(channels, 4, device='cuda', dtype=torch.bfloat16) / 2).requires_grad_()
    dy = torch.randn_like(x)

    def forward(backend: str) -> torch.Tensor:
        return causal_conv1d_backend(
            x=x,
            weight=weight,
            activation='silu',
            backend=backend,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
        )[0]

    if not cp or dist.get_rank(group=group) == 0:
        print(f'\nTokens/rank={tokens}, channels={channels}, packed={packed}', flush=True)
    _compare_cp_backends(forward=forward, x=x, weight=weight, dy=dy, group=group)

    results = []
    for backward in (False, True):
        def step(backend: str) -> None:
            y = forward(backend=backend)
            if backward:
                torch.autograd.grad(outputs=y, inputs=(x, weight), grad_outputs=dy)

        samples = _measure_cp(step=step, repeats=repeats, iterations=iterations, group=group)
        for provider, times in samples.items():
            results.append(BenchmarkResult(
                name='conv',
                parameters={'provider': provider, 'tokens_per_rank': tokens, 'channels': channels,
                            'packed': packed, 'cp': cp, 'backward': backward},
                metrics={'mean_ms': statistics.mean(times), 'median_ms': statistics.median(times)},
            ))
    return results


def _add_cp_arguments(parser):
    parser.add_argument('--tokens', type=int, nargs='+', default=[2048, 8192, 16384, 32768])
    parser.add_argument('--channels', type=int, nargs='+', default=[2048, 4096, 8192])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)


def _run_cp(args):
    if min(args.tokens) < 256 or min(args.channels + [args.repeats, args.iterations]) < 1:
        raise ValueError('tokens must be >= 256; channels, repeats and iterations must be positive')

    torch.set_num_threads(1)
    results = []
    with distributed_context(backend='nccl'):
        world_size = dist.get_world_size()
        if world_size < 2:
            raise ValueError('CP benchmark requires at least two ranks')
        if dist.get_rank() == 0:
            print(f'GPU: {torch.cuda.get_device_name()}, CP: {world_size}', flush=True)
        for tokens in args.tokens:
            for channels in args.channels:
                for packed in (False, True):
                    results.extend(_benchmark_cp_case(
                        tokens=tokens,
                        channels=channels,
                        packed=packed,
                        repeats=args.repeats,
                        iterations=args.iterations,
                        cp=True,
                    ))
    return results


def short_conv_forward(
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    cache: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_size=None,
    cp_rank=None,
    self=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x (`torch.Tensor`):
            Tensor of shape `[B, T, D]`. `B` must be 1 if `seq_idx` is provided.
        residual (`Optional[torch.Tensor]`):
            Residual tensor of shape `[B, T, D]`. Default: `None`.
        mask (`Optional[torch.Tensor]`):
            Attention mask dealing with padded positions.
        cache (`Optional[torch.Tensor]`):
            Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
            If provided, the cache is updated **inplace**.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, D, W]`. Default: `False`.
        cu_seqlens (Optional[torch.LongTensor]):
            Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
            Shape: [B+1]

    Returns:
        Tensor of shape `[B, T, D]`.
    """

    B, T, *_ = x.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if mask is not None:
        if cu_seqlens is not None:
            raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
        x = x.mul_(mask.unsqueeze(-1))

    # in decoding phase, the cache (if provided) is updated inplace
    if B * T == N:
        y, cache = self.step(
            x=x,
            residual=residual,
            cache=cache,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens
        )
        return y, cache

    # cuda backend do not support:
    # 1. both `cu_seqlens` and `cache` being provided
    # 2. both `cu_seqlens` and `output_final_state` being provided
    if self.backend == 'cuda' and (
        (cu_seqlens is not None and cache is not None) or
        (cu_seqlens is not None and output_final_state)
    ):
        self.backend = 'triton'
    # hidden_size, 1, kernel_size
    weight = self.weight.chunk(cp_size, 0)[cp_rank].squeeze(1)

    return causal_conv1d(
        x=x,
        weight=weight,
        bias=self.bias,
        residual=residual,
        initial_state=cache,
        output_final_state=output_final_state,
        activation=self.activation,
        backend=self.backend,
        cu_seqlens=cu_seqlens,
        **kwargs
    )


SCENARIOS = ('latency', 'cp')


def add_arguments(parser, scenario):
    if scenario == 'cp':
        _add_cp_arguments(parser)
    else:
        add_module_arguments(parser)


def run(args):
    if args.scenario == 'cp':
        return _run_cp(args)
    return run_module_benchmark(mark=benchmark, save_dir=args.save_dir, plot=args.plot)
