# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.utils.distributed import distributed_context
from benchmarks.utils.reporting import BenchmarkResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed import ProcessGroup


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


def add_arguments(parser):
    parser.add_argument('--tokens', type=int, nargs='+', default=[2048, 8192, 16384, 32768])
    parser.add_argument('--channels', type=int, nargs='+', default=[2048, 4096, 8192])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)


def run(args):
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
