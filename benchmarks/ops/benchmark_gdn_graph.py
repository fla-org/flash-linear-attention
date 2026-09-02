# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Benchmark eager GDN against captured CUDA Graph replay for varlen inputs."""

from __future__ import annotations

import argparse
import gc
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch

from fla.ops.gated_delta_rule import chunk_gated_delta_rule


@dataclass(frozen=True)
class Case:
    t_max: int
    actual_t: int
    n_max: int
    actual_nt: int
    nt_max: int
    cu_seqlens: tuple[int, ...]


def _git_label() -> str:
    try:
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
        commit = subprocess.check_output(['git', 'rev-parse', '--short=8', 'HEAD'], text=True).strip()
        return f'{branch}[{commit}]'
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _dtype(name: str) -> torch.dtype:
    return {'float16': torch.float16, 'bfloat16': torch.bfloat16}[name]


def _make_case(t_max: int, n_max: int, chunk_size: int, actual_ratio: float) -> Case:
    actual_t = max(n_max, round(t_max * actual_ratio))
    if actual_t > t_max:
        raise ValueError(f'actual_t={actual_t} exceeds T_max={t_max}.')

    weight_sum = n_max * (n_max + 1) // 2
    lengths = []
    assigned = 0
    for weight in range(1, n_max):
        length = max(1, actual_t * weight // weight_sum)
        lengths.append(length)
        assigned += length
    lengths.append(actual_t - assigned)
    if lengths[-1] <= 0:
        raise ValueError('actual_ratio and N_max must leave at least one token per sequence.')

    cu_seqlens = [0]
    for length in lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    actual_nt = sum((length + chunk_size - 1) // chunk_size for length in lengths)
    nt_max = (t_max + chunk_size - 1) // chunk_size + n_max - 1
    return Case(
        t_max=t_max,
        actual_t=actual_t,
        n_max=n_max,
        actual_nt=actual_nt,
        nt_max=nt_max,
        cu_seqlens=tuple(cu_seqlens),
    )


def _make_inputs(
    case: Case,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    requires_grad: bool,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device='cuda').manual_seed(42)
    tensors = {
        'q': torch.randn(1, case.t_max, heads, dim, dtype=dtype, device='cuda', generator=generator),
        'k': torch.randn(1, case.t_max, heads, dim, dtype=dtype, device='cuda', generator=generator),
        'v': torch.randn(1, case.t_max, heads, dim, dtype=dtype, device='cuda', generator=generator),
        'g': torch.randn(1, case.t_max, heads, dtype=torch.float32, device='cuda', generator=generator),
        'beta': torch.randn(1, case.t_max, heads, dtype=dtype, device='cuda', generator=generator),
        'initial_state': torch.randn(
            case.n_max,
            heads,
            dim,
            dim,
            dtype=dtype,
            device='cuda',
            generator=generator,
        ),
        'A_log': torch.randn(heads, dtype=torch.float32, device='cuda', generator=generator),
        'dt_bias': torch.randn(heads, dtype=torch.float32, device='cuda', generator=generator),
    }
    if requires_grad:
        for tensor in tensors.values():
            tensor.requires_grad_(True)
    return tensors


def _call(
    inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    use_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return chunk_gated_delta_rule(
        q=inputs['q'],
        k=inputs['k'],
        v=inputs['v'],
        g=inputs['g'],
        beta=inputs['beta'],
        initial_state=inputs['initial_state'],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=inputs['A_log'],
        dt_bias=inputs['dt_bias'],
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        use_graph=use_graph,
    )


def _warmup(step: Callable[[], object], iterations: int) -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(iterations):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def _measure(step: Callable[[], object], iterations: int, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            step()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples)


def _benchmark_forward(
    case: Case,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    chunk_size: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[float, float]:
    eager_inputs = _make_inputs(case, heads, dim, dtype, requires_grad=False)
    graph_inputs = _make_inputs(case, heads, dim, dtype, requires_grad=False)
    eager_cu = torch.tensor(case.cu_seqlens, dtype=torch.long, device='cuda')
    graph_cu = torch.tensor(case.cu_seqlens, dtype=torch.long, device='cuda')

    with torch.inference_mode():
        eager_step = partial(_call, eager_inputs, eager_cu, chunk_size, use_graph=False)
        graph_step = partial(_call, graph_inputs, graph_cu, chunk_size, use_graph=True)
        _warmup(eager_step, warmup)
        _warmup(graph_step, warmup)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = graph_step()
        replay_step = graph.replay
        _warmup(replay_step, warmup)

        eager_ms = _measure(eager_step, iterations, repeats)
        graph_ms = _measure(replay_step, iterations, repeats)

    del graph_outputs, graph
    return eager_ms, graph_ms


def _make_output_grads(
    case: Case,
    heads: int,
    dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device='cuda').manual_seed(123)
    do = torch.randn(1, case.t_max, heads, dim, dtype=dtype, device='cuda', generator=generator)
    do[:, case.actual_t:].zero_()
    dht = torch.randn(
        case.n_max,
        heads,
        dim,
        dim,
        dtype=torch.float32,
        device='cuda',
        generator=generator,
    )
    return do, dht


def _benchmark_fwdbwd(
    case: Case,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    chunk_size: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[float, float]:
    eager_inputs = _make_inputs(case, heads, dim, dtype, requires_grad=True)
    graph_inputs = _make_inputs(case, heads, dim, dtype, requires_grad=True)
    eager_cu = torch.tensor(case.cu_seqlens, dtype=torch.long, device='cuda')
    graph_cu = torch.tensor(case.cu_seqlens, dtype=torch.long, device='cuda')
    eager_do, eager_dht = _make_output_grads(case, heads, dim, dtype)
    graph_do, graph_dht = _make_output_grads(case, heads, dim, dtype)

    def step(
        inputs: dict[str, torch.Tensor],
        cu_seqlens: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        use_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for tensor in inputs.values():
            if tensor.grad is not None:
                tensor.grad.zero_()
        o, ht = _call(inputs, cu_seqlens, chunk_size, use_graph=use_graph)
        torch.autograd.backward((o, ht), (do, dht))
        return o, ht

    eager_step = partial(step, eager_inputs, eager_cu, eager_do, eager_dht, use_graph=False)
    graph_step = partial(step, graph_inputs, graph_cu, graph_do, graph_dht, use_graph=True)
    _warmup(eager_step, warmup)
    _warmup(graph_step, warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = graph_step()
    replay_step = graph.replay
    _warmup(replay_step, warmup)

    eager_ms = _measure(eager_step, iterations, repeats)
    graph_ms = _measure(replay_step, iterations, repeats)
    del graph_outputs, graph
    return eager_ms, graph_ms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t-max', type=int, nargs='+', default=[256, 512, 1024, 2048])
    parser.add_argument('--n-max', type=int, default=4)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--chunk-size', type=int, choices=(16, 32, 64), default=64)
    parser.add_argument('--actual-ratio', type=float, default=0.75)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--modes', nargs='+', choices=('fwd', 'fwdbwd'), default=['fwd', 'fwdbwd'])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires an NVIDIA CUDA device.')
    if not 0 < args.actual_ratio <= 1:
        raise ValueError('actual_ratio must be in (0, 1].')
    if min(args.t_max) < args.n_max:
        raise ValueError('Every T_max must be at least N_max.')
    if min(args.heads, args.dim, args.n_max, args.warmup, args.iterations, args.repeats) < 1:
        raise ValueError('Heads, dim, N_max, warmup, iterations, and repeats must be positive.')

    dtype = _dtype(args.dtype)
    print('=' * 112)
    print(f'Machine: {torch.cuda.get_device_name()} | CUDA {torch.version.cuda} | PyTorch {torch.__version__}')
    print(f'Ref: {_git_label()} | H={args.heads} | D={args.dim} | BT={args.chunk_size} | dtype={args.dtype}')
    print(f'Warmup={args.warmup} | iterations={args.iterations} | repeats={args.repeats}')
    print('=' * 112)
    print(f"{'mode':<8} {'T_max':>7} {'actual_t':>9} {'N_max':>7} {'actual_nt':>10} {'NT_max':>8} "
          f"{'eager(ms)':>12} {'graph(ms)':>12} {'speedup':>9}")
    print('-' * 112)

    for mode in args.modes:
        benchmark = _benchmark_forward if mode == 'fwd' else _benchmark_fwdbwd
        for t_max in args.t_max:
            case = _make_case(t_max, args.n_max, args.chunk_size, args.actual_ratio)
            eager_ms, graph_ms = benchmark(
                case=case,
                heads=args.heads,
                dim=args.dim,
                dtype=dtype,
                chunk_size=args.chunk_size,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            print(
                f'{mode:<8} {case.t_max:>7} {case.actual_t:>9} {case.n_max:>7} {case.actual_nt:>10} '
                f'{case.nt_max:>8} {eager_ms:>12.3f} {graph_ms:>12.3f} {eager_ms / graph_ms:>8.2f}x',
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
    print('=' * 112)
    print('CUDA Graph capture, JIT compilation, autotuning, and warmup are excluded from reported latency.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
