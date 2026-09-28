# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Measure GDN CUDA Graph speedups with capacity, token, and input-update costs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import triton

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.kda import KimiDeltaAttention
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda import chunk_kda
from fla.ops.utils.graph import route_graph_execution
from fla.ops.utils.index import prepare_chunk_indices_static
from fla.utils import assert_close

TOKEN_INPUTS = ('q', 'k', 'v', 'g', 'beta')
LAYOUTS = ('balanced', 'ragged', 'packed')


@dataclass(frozen=True)
class Case:
    t_max: int
    n_max: int
    actual_t: int
    actual_nt: int
    nt_max: int
    layout: str
    cu_seqlens: tuple[int, ...]


@dataclass(frozen=True)
class Timing:
    wall_ms: float
    gpu_ms: float
    wall_min_ms: float
    wall_max_ms: float
    wall_p95_ms: float


@dataclass
class ComponentBundle:
    owner: object | None
    tensors: dict[str, torch.Tensor]


def _git_label() -> str:
    try:
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
        commit = subprocess.check_output(['git', 'rev-parse', '--short=8', 'HEAD'], text=True).strip()
        return f'{branch}[{commit}]'
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _dtype(name: str) -> torch.dtype:
    return {'float16': torch.float16, 'bfloat16': torch.bfloat16}[name]


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _make_lengths(actual_t: int, n_max: int, layout: str) -> list[int]:
    if actual_t < 1:
        raise ValueError('actual_t must be positive.')
    active = min(actual_t, n_max)
    if layout == 'packed':
        return [actual_t] + [0] * (n_max - 1)

    if layout == 'balanced':
        base, remainder = divmod(actual_t, active)
        lengths = [base + (index < remainder) for index in range(active)]
    else:
        weights = list(range(1, active + 1))
        weight_sum = sum(weights)
        lengths = [max(1, actual_t * weight // weight_sum) for weight in weights]
        while sum(lengths) > actual_t:
            for index in reversed(range(active)):
                if lengths[index] > 1 and sum(lengths) > actual_t:
                    lengths[index] -= 1
        while sum(lengths) < actual_t:
            lengths[-1] += 1

    return lengths + [0] * (n_max - active)


def _make_case(t_max: int, n_max: int, actual_t: int, chunk_size: int, layout: str) -> Case:
    if actual_t > t_max:
        raise ValueError(f'actual_t={actual_t} exceeds T_max={t_max}.')
    lengths = _make_lengths(actual_t, n_max, layout)
    cu_seqlens = [0]
    for length in lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    actual_nt = sum(_ceil_div(length, chunk_size) for length in lengths if length)
    nt_max = _ceil_div(t_max, chunk_size) + n_max - 1
    return Case(t_max, n_max, actual_t, actual_nt, nt_max, layout, tuple(cu_seqlens))


def _make_inputs(
    case: Case,
    heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
    seed: int,
    requires_grad: bool,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device='cuda').manual_seed(seed)
    inputs = {
        'q': torch.randn(1, case.t_max, heads, key_dim, dtype=dtype, device='cuda', generator=generator) * 0.4,
        'k': torch.randn(1, case.t_max, heads, key_dim, dtype=dtype, device='cuda', generator=generator) * 0.4,
        'v': torch.randn(1, case.t_max, value_heads, value_dim, dtype=dtype, device='cuda', generator=generator) * 0.1,
        'g': torch.randn(1, case.t_max, value_heads, dtype=torch.float32, device='cuda', generator=generator) * 0.2,
        'beta': torch.randn(1, case.t_max, value_heads, dtype=dtype, device='cuda', generator=generator) * 0.4,
        'initial_state': torch.randn(
            case.n_max,
            value_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device='cuda',
            generator=generator,
        ) * 0.03,
        'A_log': -0.7 + torch.rand(value_heads, dtype=torch.float32, device='cuda', generator=generator) * 0.2,
        'dt_bias': torch.randn(value_heads, dtype=torch.float32, device='cuda', generator=generator) * 0.1,
    }
    if requires_grad:
        for value in inputs.values():
            value.requires_grad_(True)
    return inputs


def _slice_inputs(inputs: dict[str, torch.Tensor], actual_t: int, requires_grad: bool) -> dict[str, torch.Tensor]:
    sliced = {
        name: value[:, :actual_t].detach().clone() if name in TOKEN_INPUTS else value.detach().clone()
        for name, value in inputs.items()
    }
    if requires_grad:
        for value in sliced.values():
            value.requires_grad_(True)
    return sliced


def _make_output_grads(
    case: Case,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device='cuda').manual_seed(seed)
    do = torch.randn(1, case.t_max, value_heads, value_dim, dtype=dtype, device='cuda', generator=generator) * 0.2
    do[:, case.actual_t:].zero_()
    dht = torch.randn(
        case.n_max,
        value_heads,
        value_dim,
        key_dim,
        dtype=torch.float32,
        device='cuda',
        generator=generator,
    ) * 0.2
    return do, dht


def _call(
    inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    use_graph: bool,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    graph_t_max: int | None = None,
    graph_n_max: int | None = None,
    graph_nt_max: int | None = None,
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
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        graph_t_max=graph_t_max,
        graph_n_max=graph_n_max,
        graph_nt_max=graph_nt_max,
    )


def _zero_grads(inputs: dict[str, torch.Tensor]) -> None:
    for value in inputs.values():
        if value.grad is not None:
            value.grad.zero_()


def _backward_step(
    inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    chunk_size: int,
    use_graph: bool,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    graph_t_max: int | None = None,
    graph_n_max: int | None = None,
    graph_nt_max: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _zero_grads(inputs)
    output, final_state = _call(
        inputs,
        cu_seqlens,
        chunk_size,
        use_graph,
        chunk_indices,
        chunk_offsets,
        graph_t_max,
        graph_n_max,
        graph_nt_max,
    )
    torch.autograd.backward((output, final_state), (do, dht))
    return output, final_state


def _copy_inputs(dst: dict[str, torch.Tensor], src: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name in dst:
            dst[name].copy_(src[name])


def _graph_update_step(
    graph: torch.cuda.CUDAGraph,
    static_inputs: dict[str, torch.Tensor],
    source_inputs: dict[str, torch.Tensor],
    static_cu: torch.Tensor,
    source_cu: torch.Tensor,
    static_do: torch.Tensor | None,
    source_do: torch.Tensor | None,
    static_dht: torch.Tensor | None,
    source_dht: torch.Tensor | None,
    static_indices: torch.Tensor,
    source_indices: torch.Tensor,
    static_offsets: torch.Tensor,
    source_offsets: torch.Tensor,
) -> None:
    _copy_step(
        static_inputs,
        source_inputs,
        static_cu,
        source_cu,
        static_do,
        source_do,
        static_dht,
        source_dht,
        static_indices,
        source_indices,
        static_offsets,
        source_offsets,
    )
    graph.replay()


def _copy_step(
    static_inputs: dict[str, torch.Tensor],
    source_inputs: dict[str, torch.Tensor],
    static_cu: torch.Tensor,
    source_cu: torch.Tensor,
    static_do: torch.Tensor | None,
    source_do: torch.Tensor | None,
    static_dht: torch.Tensor | None,
    source_dht: torch.Tensor | None,
    static_indices: torch.Tensor | None = None,
    source_indices: torch.Tensor | None = None,
    static_offsets: torch.Tensor | None = None,
    source_offsets: torch.Tensor | None = None,
) -> None:
    _copy_inputs(static_inputs, source_inputs)
    static_cu.copy_(source_cu)
    if static_do is not None:
        static_do.copy_(source_do)
        static_dht.copy_(source_dht)
    if static_indices is not None:
        static_indices.copy_(source_indices)
    if static_offsets is not None:
        static_offsets.copy_(source_offsets)


def _warmup(step: Callable[[], object], iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def _measure(step: Callable[[], object], iterations: int, repeats: int) -> Timing:
    wall_samples = []
    gpu_samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(iterations):
            step()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000 / iterations)
        gpu_samples.append(start.elapsed_time(end) / iterations)
    return Timing(
        wall_ms=statistics.median(wall_samples),
        gpu_ms=statistics.median(gpu_samples),
        wall_min_ms=min(wall_samples),
        wall_max_ms=max(wall_samples),
        wall_p95_ms=statistics.quantiles(wall_samples, n=20, method='inclusive')[-1]
        if len(wall_samples) > 1 else wall_samples[0],
    )


def _capture(step: Callable[[], object]) -> tuple[torch.cuda.CUDAGraph, object, float]:
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    start = time.perf_counter()
    with torch.cuda.graph(graph):
        outputs = step()
    torch.cuda.synchronize()
    return graph, outputs, (time.perf_counter() - start) * 1000


def _speedup(baseline: Timing, candidate: Timing) -> float:
    return baseline.wall_ms / candidate.wall_ms


def _break_even(capture_ms: float, baseline: Timing, candidate: Timing) -> int | None:
    savings = baseline.wall_ms - candidate.wall_ms
    return math.ceil(capture_ms / savings) if savings > 0 else None


def _actual_sequences(case: Case) -> int:
    return sum(eos > bos for bos, eos in zip(case.cu_seqlens[:-1], case.cu_seqlens[1:], strict=True))


def _static_metadata(cu_seqlens: torch.Tensor, chunk_size: int, nt_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    return prepare_chunk_indices_static(cu_seqlens, chunk_size, nt_max)


def _assert_finite(name: str, value: torch.Tensor | None) -> None:
    if value is not None:
        assert torch.isfinite(value).all(), f'{name} contains non-finite values'


def _assert_operator_smoke(
    graph_outputs: tuple[torch.Tensor, torch.Tensor],
    eager_outputs: tuple[torch.Tensor, torch.Tensor],
    actual_t: int,
    graph_inputs: dict[str, torch.Tensor],
    eager_inputs: dict[str, torch.Tensor],
    requires_grad: bool,
) -> None:
    graph_output, graph_state = graph_outputs
    eager_output, eager_state = eager_outputs
    _assert_finite('graph output', graph_output)
    _assert_finite('graph final state', graph_state)
    _assert_finite('eager output', eager_output)
    _assert_finite('eager final state', eager_state)
    torch.testing.assert_close(eager_output, graph_output[:, :actual_t], atol=0.05, rtol=0.05)
    torch.testing.assert_close(eager_state, graph_state, atol=0.05, rtol=0.05)
    if actual_t < graph_output.shape[1]:
        torch.testing.assert_close(graph_output[:, actual_t:], torch.zeros_like(graph_output[:, actual_t:]))
    if requires_grad:
        for name, eager_value in eager_inputs.items():
            if eager_value.grad is None:
                continue
            graph_value = graph_inputs[name].grad
            assert graph_value is not None, f'missing graph gradient for {name}'
            compared = graph_value[:, :actual_t] if name in TOKEN_INPUTS else graph_value
            expected = eager_value.grad[:, :actual_t] if name in TOKEN_INPUTS else eager_value.grad
            torch.testing.assert_close(expected, compared, atol=0.08, rtol=0.08)
            _assert_finite(f'graph d{name}', graph_value)
        if actual_t < graph_output.shape[1]:
            for name in TOKEN_INPUTS:
                graph_grad = graph_inputs[name].grad
                assert graph_grad is not None
                torch.testing.assert_close(graph_grad[:, actual_t:], torch.zeros_like(graph_grad[:, actual_t:]))


def _component_decision(case: Case, args: argparse.Namespace):
    return route_graph_execution(
        'auto',
        actual_tokens=case.actual_t,
        actual_sequences=_actual_sequences(case),
        actual_nt=case.actual_nt,
        t_max=case.t_max,
        n_max=case.n_max,
        nt_max=case.nt_max,
        min_graph_utilization=args.min_graph_utilization,
        chunk_size=args.chunk_size,
    )


def _component_row(
    component: str,
    case: Case,
    decision,
    eager_live: Timing,
    eager_fixed: Timing,
    graph_replay: Timing,
    graph_update: Timing,
    copy_only: Timing,
    capture_ms: float,
    graph_warmup_ms: float,
    replay_warmup_ms: float,
    peak_gib: float,
    mode: str = 'fwd',
) -> dict[str, object]:
    auto = graph_update if decision.selected_path == 'graph' else eager_live
    return {
        'component': component,
        'mode': mode,
        'scenario': case.layout,
        'route': decision.selected_path,
        'route_reason': decision.reason,
        'T_max': case.t_max,
        'N_max': case.n_max,
        'NT_max': case.nt_max,
        'actual_t': case.actual_t,
        'actual_sequences': decision.actual_sequences,
        'actual_NT': case.actual_nt,
        'utilization': decision.utilization,
        'eager_ms': eager_live.wall_ms,
        'eager_gpu_ms': eager_live.gpu_ms,
        'eager_p95_ms': eager_live.wall_p95_ms,
        'eager_fixed_ms': eager_fixed.wall_ms,
        'eager_fixed_gpu_ms': eager_fixed.gpu_ms,
        'eager_fixed_p95_ms': eager_fixed.wall_p95_ms,
        'graph_replay_ms': graph_replay.wall_ms,
        'graph_replay_gpu_ms': graph_replay.gpu_ms,
        'graph_replay_p95_ms': graph_replay.wall_p95_ms,
        'graph_update_ms': graph_update.wall_ms,
        'graph_update_gpu_ms': graph_update.gpu_ms,
        'graph_update_p95_ms': graph_update.wall_p95_ms,
        'copy_only_ms': copy_only.wall_ms,
        'copy_only_gpu_ms': copy_only.gpu_ms,
        'copy_only_p95_ms': copy_only.wall_p95_ms,
        'auto_ms': auto.wall_ms,
        'auto_p95_ms': auto.wall_p95_ms,
        'speedup_replay': _speedup(eager_live, graph_replay),
        'speedup_update': _speedup(eager_live, graph_update),
        'speedup_auto': _speedup(eager_live, auto),
        'speedup_fixed_replay': _speedup(eager_fixed, graph_replay),
        'speedup_fixed_update': _speedup(eager_fixed, graph_update),
        'speedup_auto_vs_fixed': _speedup(eager_fixed, auto),
        'capture_ms': capture_ms,
        'graph_warmup_ms': graph_warmup_ms,
        'replay_warmup_ms': replay_warmup_ms,
        'break_even_replays': _break_even(capture_ms, eager_fixed, graph_update),
        'peak_allocated_gib': peak_gib,
        'graph_capture_attempted': True,
        'graph_buffers_created': True,
    }


def _print_component_row(row: dict[str, object]) -> None:
    print(
        f"{row['component']:<10} {row['scenario']:<8} {row['route']:<6} "
        f"{row['T_max']:>6} {row['N_max']:>5} {row['NT_max']:>7} {row['actual_t']:>8} "
        f"{row['actual_NT']:>9} {row['utilization']:>6.2f} {row['eager_ms']:>10.3f} "
        f"{row['graph_replay_ms']:>10.3f} {row['graph_update_ms']:>10.3f} {row['copy_only_ms']:>9.3f} "
        f"{row['speedup_auto']:>8.2f}x {row['capture_ms']:>9.1f} "
        f"{str(row['break_even_replays'] or '-'):>5} {row['peak_allocated_gib']:>7.2f}",
        flush=True,
    )


def _component_output_state(result: object) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(result, tuple):
        output = result[0]
        state = result[1] if len(result) > 1 and isinstance(result[1], torch.Tensor) else None
    else:
        output, state = result, None
    assert isinstance(output, torch.Tensor)
    return output, state


def _set_component_requires_grad(bundle: ComponentBundle, enabled: bool) -> None:
    bundle.tensors = {
        name: value.detach().requires_grad_(enabled)
        for name, value in bundle.tensors.items()
    }
    if bundle.owner is not None and hasattr(bundle.owner, 'parameters'):
        for parameter in bundle.owner.parameters():
            parameter.requires_grad_(enabled)


def _zero_component_grads(bundle: ComponentBundle) -> None:
    for value in bundle.tensors.values():
        if value.grad is not None:
            value.grad.zero_()
    if bundle.owner is not None and hasattr(bundle.owner, 'parameters'):
        for parameter in bundle.owner.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()


def _component_grads(bundle: ComponentBundle) -> dict[str, torch.Tensor]:
    gradients = {}
    for name, value in bundle.tensors.items():
        if value.grad is not None:
            gradients[f'tensor:{name}'] = value.grad
    if bundle.owner is not None and hasattr(bundle.owner, 'named_parameters'):
        for name, parameter in bundle.owner.named_parameters():
            if parameter.grad is not None:
                gradients[f'parameter:{name}'] = parameter.grad
    return gradients


def _component_output_grads(
    result: object,
    actual_t: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output, state = _component_output_state(result)
    output_grad = torch.randn_like(output)
    if output.ndim >= 2 and actual_t < output.shape[1]:
        output_grad[:, actual_t:] = 0
    state_grad = torch.randn_like(state) if state is not None else None
    return output_grad, state_grad


def _component_backward_step(
    call_bundle: Callable[..., object],
    bundle: ComponentBundle,
    cu_seqlens: torch.Tensor,
    output_grad: torch.Tensor,
    state_grad: torch.Tensor | None,
    use_graph: bool,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    graph_t_max: int | None = None,
    graph_n_max: int | None = None,
    graph_nt_max: int | None = None,
) -> object:
    _zero_component_grads(bundle)
    result = call_bundle(
        bundle,
        cu_seqlens,
        use_graph,
        chunk_indices,
        chunk_offsets,
        graph_t_max,
        graph_n_max,
        graph_nt_max,
    )
    output, state = _component_output_state(result)
    if state is None:
        torch.autograd.backward(output, output_grad)
    else:
        assert state_grad is not None
        torch.autograd.backward((output, state), (output_grad, state_grad))
    return result


def _copy_component_bundle(destination: ComponentBundle, source: ComponentBundle) -> None:
    with torch.no_grad():
        for name, value in destination.tensors.items():
            value.copy_(source.tensors[name])


def _component_gradient_view(name: str, gradient: torch.Tensor, actual_t: int) -> torch.Tensor:
    token_names = {'x', 'q', 'k', 'v', 'g', 'beta', 'residual'}
    return gradient[:, :actual_t] if name in token_names else gradient


def _error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    difference = reference - candidate
    max_abs = difference.abs().max().item()
    rms_ratio = difference.square().mean().sqrt().item() / (reference.square().mean().sqrt().item() + 1e-8)
    return max_abs, rms_ratio


def _assert_component_smoke(
    graph_result: object,
    eager_result: object,
    actual_t: int,
) -> None:
    graph_output, graph_state = _component_output_state(graph_result)
    eager_output, eager_state = _component_output_state(eager_result)
    _assert_finite('graph component output', graph_output)
    _assert_finite('eager component output', eager_output)
    torch.testing.assert_close(eager_output, graph_output[:, :actual_t], atol=0.06, rtol=0.06)
    if graph_state is not None or eager_state is not None:
        assert graph_state is not None and eager_state is not None
        _assert_finite('graph component state', graph_state)
        _assert_finite('eager component state', eager_state)
        torch.testing.assert_close(eager_state, graph_state, atol=0.06, rtol=0.06)
    if actual_t < graph_output.shape[1]:
        torch.testing.assert_close(graph_output[:, actual_t:], torch.zeros_like(graph_output[:, actual_t:]))


@torch.inference_mode()
def _run_component_benchmark(
    args: argparse.Namespace,
    component: str,
    cases: list[Case],
    make_bundle: Callable[[Case, int], ComponentBundle],
    clone_bundle: Callable[[ComponentBundle, int], ComponentBundle],
    call_bundle: Callable[..., object],
) -> list[dict[str, object]]:
    """Benchmark one component with a fixed graph bucket."""
    capacity_case = cases[0]
    static_bundle = make_bundle(capacity_case, args.seed)
    static_cu = torch.tensor(capacity_case.cu_seqlens, dtype=torch.long, device='cuda')
    static_indices, static_offsets = _static_metadata(static_cu, args.chunk_size, capacity_case.nt_max)

    def graph_step(
        static_bundle=static_bundle,
        static_cu=static_cu,
        static_indices=static_indices,
        static_offsets=static_offsets,
    ):
        return call_bundle(
            static_bundle,
            static_cu,
            True,
            static_indices,
            static_offsets,
            capacity_case.t_max,
            capacity_case.n_max,
            capacity_case.nt_max,
        )

    graph_warmup_ms = _warmup(graph_step, args.warmup)
    graph, graph_output, capture_ms = _capture(graph_step)
    replay_warmup_ms = _warmup(graph.replay, args.warmup)
    rows = []

    for case in cases:
        # Keep tensor values and layer weights fixed while changing only the
        # variable-length metadata for each layout.
        source_bundle = clone_bundle(static_bundle, case.t_max)
        eager_fixed_bundle = clone_bundle(source_bundle, case.t_max)
        eager_live_bundle = clone_bundle(source_bundle, case.actual_t)
        source_cu = torch.tensor(case.cu_seqlens, dtype=torch.long, device='cuda')
        source_indices, source_offsets = _static_metadata(source_cu, args.chunk_size, case.nt_max)
        eager_fixed_cu = source_cu.clone()
        eager_live_cu = source_cu.clone()

        eager_fixed_step = partial(call_bundle, eager_fixed_bundle, eager_fixed_cu, False, None, None, None, None, None)
        eager_live_step = partial(call_bundle, eager_live_bundle, eager_live_cu, False, None, None, None, None, None)

        def copy_step(
            static_bundle=static_bundle,
            source_bundle=source_bundle,
            static_cu=static_cu,
            source_cu=source_cu,
            static_indices=static_indices,
            source_indices=source_indices,
            static_offsets=static_offsets,
            source_offsets=source_offsets,
        ):
            for name, destination in static_bundle.tensors.items():
                destination.copy_(source_bundle.tensors[name])
            static_cu.copy_(source_cu)
            static_indices.copy_(source_indices)
            static_offsets.copy_(source_offsets)

        def graph_update_step(copy_step=copy_step, graph=graph):
            copy_step()
            graph.replay()

        copy_step()
        graph.replay()
        torch.cuda.synchronize()
        eager_smoke = eager_live_step()
        torch.cuda.synchronize()
        _assert_component_smoke(graph_output, eager_smoke, case.actual_t)
        metadata_ptrs = (static_cu.data_ptr(), static_indices.data_ptr(), static_offsets.data_ptr())

        torch.cuda.reset_peak_memory_stats()
        _warmup(eager_fixed_step, args.warmup)
        _warmup(eager_live_step, args.warmup)
        _warmup(copy_step, args.warmup)
        _warmup(graph_update_step, args.warmup)
        eager_live = _measure(eager_live_step, args.iterations, args.repeats)
        eager_fixed = _measure(eager_fixed_step, args.iterations, args.repeats)
        graph_replay = _measure(graph.replay, args.iterations, args.repeats)
        graph_update = _measure(graph_update_step, args.iterations, args.repeats)
        copy_only = _measure(copy_step, args.iterations, args.repeats)
        decision = _component_decision(case, args)
        row = _component_row(
            component,
            case,
            decision,
            eager_live,
            eager_fixed,
            graph_replay,
            graph_update,
            copy_only,
            capture_ms,
            graph_warmup_ms,
            replay_warmup_ms,
            torch.cuda.max_memory_allocated() / (1024 ** 3),
        )
        row['current_allocated_gib'] = torch.cuda.memory_allocated() / (1024 ** 3)
        row['current_reserved_gib'] = torch.cuda.memory_reserved() / (1024 ** 3)
        row['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / (1024 ** 3)
        row['metadata_ptr_stable'] = metadata_ptrs == (
            static_cu.data_ptr(), static_indices.data_ptr(), static_offsets.data_ptr())
        row['correctness_smoke'] = True
        row['auto_semantics'] = 'external_scheduler: graph_update (copy+replay) or eager_live'
        rows.append(row)
        _print_component_row(row)
        del source_bundle, eager_fixed_bundle, eager_live_bundle, source_cu, eager_fixed_cu, eager_live_cu
        gc.collect()

    del graph_output, graph, static_bundle, static_cu
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return rows


def _run_component_backward_benchmark(
    args: argparse.Namespace,
    component: str,
    cases: list[Case],
    make_bundle: Callable[[Case, int], ComponentBundle],
    clone_bundle: Callable[[ComponentBundle, int], ComponentBundle],
    call_bundle: Callable[..., object],
) -> list[dict[str, object]]:
    """Benchmark one component with forward and backward captured together."""
    rows = []
    with torch.enable_grad():
        capacity_case = cases[0]
        static_bundle = make_bundle(capacity_case, args.seed)
        _set_component_requires_grad(static_bundle, True)
        static_cu = torch.tensor(capacity_case.cu_seqlens, dtype=torch.long, device='cuda')
        static_indices, static_offsets = _static_metadata(static_cu, args.chunk_size, capacity_case.nt_max)

        probe_result = call_bundle(
            static_bundle,
            static_cu,
            True,
            static_indices,
            static_offsets,
            capacity_case.t_max,
            capacity_case.n_max,
            capacity_case.nt_max,
        )
        static_do, static_dht = _component_output_grads(probe_result, capacity_case.actual_t)
        del probe_result
        _zero_component_grads(static_bundle)

        def warmup_step(
            call_bundle=call_bundle,
            static_bundle=static_bundle,
            static_cu=static_cu,
            static_do=static_do,
            static_dht=static_dht,
            static_indices=static_indices,
            static_offsets=static_offsets,
            capacity_case=capacity_case,
        ) -> None:
            result = _component_backward_step(
                call_bundle,
                static_bundle,
                static_cu,
                static_do,
                static_dht,
                True,
                static_indices,
                static_offsets,
                capacity_case.t_max,
                capacity_case.n_max,
                capacity_case.nt_max,
            )
            del result

        graph_warmup_ms = _warmup(warmup_step, args.warmup)
        _zero_component_grads(static_bundle)

        def capture_step(
            call_bundle=call_bundle,
            static_bundle=static_bundle,
            static_cu=static_cu,
            static_do=static_do,
            static_dht=static_dht,
            static_indices=static_indices,
            static_offsets=static_offsets,
            capacity_case=capacity_case,
        ) -> object:
            return _component_backward_step(
                call_bundle,
                static_bundle,
                static_cu,
                static_do,
                static_dht,
                True,
                static_indices,
                static_offsets,
                capacity_case.t_max,
                capacity_case.n_max,
                capacity_case.nt_max,
            )

        graph, graph_result, capture_ms = _capture(capture_step)
        replay_warmup_ms = _warmup(graph.replay, args.warmup)
        metadata_ptrs = (static_cu.data_ptr(), static_indices.data_ptr(), static_offsets.data_ptr())

        for case_index, case in enumerate(cases):
            source_bundle = clone_bundle(static_bundle, case.t_max)
            eager_fixed_bundle = clone_bundle(source_bundle, case.t_max)
            eager_live_bundle = clone_bundle(source_bundle, case.actual_t)
            _set_component_requires_grad(eager_fixed_bundle, True)
            _set_component_requires_grad(eager_live_bundle, True)
            source_cu = torch.tensor(case.cu_seqlens, dtype=torch.long, device='cuda')
            source_indices, source_offsets = _static_metadata(source_cu, args.chunk_size, case.nt_max)
            eager_fixed_cu = source_cu.clone()
            eager_live_cu = source_cu.clone()

            source_do = torch.randn_like(static_do)
            if case.actual_t < source_do.shape[1]:
                source_do[:, case.actual_t:] = 0
            source_dht = torch.randn_like(static_dht) if static_dht is not None else None
            eager_fixed_do = source_do.clone()
            eager_fixed_dht = source_dht.clone() if source_dht is not None else None
            eager_live_do = source_do[:, :case.actual_t].clone()
            eager_live_dht = source_dht.clone() if source_dht is not None else None

            with torch.no_grad():
                _copy_component_bundle(static_bundle, source_bundle)
                static_cu.copy_(source_cu)
                static_indices.copy_(source_indices)
                static_offsets.copy_(source_offsets)
                static_do.copy_(source_do)
                if static_dht is not None:
                    assert source_dht is not None
                    static_dht.copy_(source_dht)
            graph.replay()
            torch.cuda.synchronize()
            graph_grads = {
                key: gradient.detach().clone()
                for key, gradient in _component_grads(static_bundle).items()
            }

            eager_result = _component_backward_step(
                call_bundle,
                eager_live_bundle,
                eager_live_cu,
                eager_live_do,
                eager_live_dht,
                False,
            )
            torch.cuda.synchronize()
            graph_output, graph_state = _component_output_state(graph_result)
            eager_output, eager_state = _component_output_state(eager_result)
            _assert_finite('graph component output', graph_output)
            _assert_finite('graph component state', graph_state)
            _assert_finite('eager component output', eager_output)
            _assert_finite('eager component state', eager_state)
            torch.testing.assert_close(eager_output, graph_output[:, :case.actual_t], atol=0.08, rtol=0.08)
            if graph_state is not None or eager_state is not None:
                assert graph_state is not None and eager_state is not None
                torch.testing.assert_close(eager_state, graph_state, atol=0.08, rtol=0.08)
            if case.actual_t < graph_output.shape[1]:
                torch.testing.assert_close(graph_output[:, case.actual_t:], torch.zeros_like(graph_output[:, case.actual_t:]))

            eager_grads = _component_grads(eager_live_bundle)
            current_graph_grads = _component_grads(static_bundle)
            assert graph_grads.keys() == current_graph_grads.keys() == eager_grads.keys()
            for key in graph_grads:
                torch.testing.assert_close(graph_grads[key], current_graph_grads[key], atol=0, rtol=0)
            gradient_ratio_limit = {'layer': 0.04, 'conv': 0.03, 'kda': 0.04, 'kda_layer': 0.05}[component]
            gradient_max_abs = 0.0
            gradient_max_rms_ratio = 0.0
            for key in graph_grads:
                tensor_name = key.split(':', 1)[1]
                graph_gradient = graph_grads[key]
                if key.startswith('tensor:'):
                    graph_gradient = _component_gradient_view(tensor_name, graph_gradient, case.actual_t)
                max_abs, rms_ratio = _error_metrics(eager_grads[key], graph_gradient)
                gradient_max_abs = max(gradient_max_abs, max_abs)
                gradient_max_rms_ratio = max(gradient_max_rms_ratio, rms_ratio)
                assert_close(
                    f'{component} live {key}',
                    eager_grads[key],
                    graph_gradient,
                    gradient_ratio_limit,
                )
                _assert_finite(f'graph {key}', graph_grads[key])
            if case.actual_t < graph_output.shape[1]:
                for name in ('x', 'q', 'k', 'v', 'g', 'beta', 'residual'):
                    key = f'tensor:{name}'
                    if key in graph_grads:
                        tail = graph_grads[key][:, case.actual_t:]
                        torch.testing.assert_close(tail, torch.zeros_like(tail))

            def eager_fixed_step(
                call_bundle=call_bundle,
                eager_fixed_bundle=eager_fixed_bundle,
                eager_fixed_cu=eager_fixed_cu,
                eager_fixed_do=eager_fixed_do,
                eager_fixed_dht=eager_fixed_dht,
            ) -> object:
                return _component_backward_step(
                    call_bundle,
                    eager_fixed_bundle,
                    eager_fixed_cu,
                    eager_fixed_do,
                    eager_fixed_dht,
                    False,
                )

            def eager_live_step(
                call_bundle=call_bundle,
                eager_live_bundle=eager_live_bundle,
                eager_live_cu=eager_live_cu,
                eager_live_do=eager_live_do,
                eager_live_dht=eager_live_dht,
            ) -> object:
                return _component_backward_step(
                    call_bundle,
                    eager_live_bundle,
                    eager_live_cu,
                    eager_live_do,
                    eager_live_dht,
                    False,
                )

            def copy_step(
                static_bundle=static_bundle,
                source_bundle=source_bundle,
                static_cu=static_cu,
                source_cu=source_cu,
                static_indices=static_indices,
                source_indices=source_indices,
                static_offsets=static_offsets,
                source_offsets=source_offsets,
                static_do=static_do,
                source_do=source_do,
                static_dht=static_dht,
                source_dht=source_dht,
            ) -> None:
                with torch.no_grad():
                    _copy_component_bundle(static_bundle, source_bundle)
                    static_cu.copy_(source_cu)
                    static_indices.copy_(source_indices)
                    static_offsets.copy_(source_offsets)
                    static_do.copy_(source_do)
                    if static_dht is not None:
                        assert source_dht is not None
                        static_dht.copy_(source_dht)

            def graph_update_step(copy_step=copy_step, graph=graph) -> None:
                copy_step()
                graph.replay()

            torch.cuda.reset_peak_memory_stats()
            _warmup(eager_fixed_step, args.warmup)
            _warmup(eager_live_step, args.warmup)
            _warmup(graph.replay, args.warmup)
            _warmup(graph_update_step, args.warmup)
            _warmup(copy_step, args.warmup)
            eager_live = _measure(eager_live_step, args.iterations, args.repeats)
            eager_fixed = _measure(eager_fixed_step, args.iterations, args.repeats)
            graph_replay = _measure(graph.replay, args.iterations, args.repeats)
            graph_update = _measure(graph_update_step, args.iterations, args.repeats)
            copy_only = _measure(copy_step, args.iterations, args.repeats)
            decision = _component_decision(case, args)
            row = _component_row(
                component,
                case,
                decision,
                eager_live,
                eager_fixed,
                graph_replay,
                graph_update,
                copy_only,
                capture_ms,
                graph_warmup_ms,
                replay_warmup_ms,
                torch.cuda.max_memory_allocated() / (1024 ** 3),
                mode='fwdbwd',
            )
            row['metadata_ptr_stable'] = metadata_ptrs == (
                static_cu.data_ptr(), static_indices.data_ptr(), static_offsets.data_ptr()
            )
            row['correctness_smoke'] = True
            row['correctness_backward'] = True
            row['correctness_oracle'] = 'eager_live output/state/gradients'
            row['gradient_ratio_limit'] = gradient_ratio_limit
            row['gradient_max_abs'] = gradient_max_abs
            row['gradient_max_rms_ratio'] = gradient_max_rms_ratio
            row['auto_semantics'] = 'external_scheduler: graph_update (copy+replay) or eager_live'
            row['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / (1024 ** 3)
            row['current_allocated_gib'] = torch.cuda.memory_allocated() / (1024 ** 3)
            row['current_reserved_gib'] = torch.cuda.memory_reserved() / (1024 ** 3)
            rows.append(row)
            _print_component_row(row)
            del eager_result, source_bundle, eager_fixed_bundle, eager_live_bundle
            del source_cu, source_indices, source_offsets, eager_fixed_cu, eager_live_cu
            del source_do, eager_fixed_do, eager_live_do
            if source_dht is not None:
                del source_dht, eager_fixed_dht, eager_live_dht
            gc.collect()

        del graph_result, graph, static_bundle, static_cu, static_indices, static_offsets, static_do
        if static_dht is not None:
            del static_dht
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return rows


def _layer_dimensions(args: argparse.Namespace) -> tuple[int, int, int, float]:
    hidden_size = args.layer_hidden_size or args.heads * args.key_dim
    head_dim = args.layer_head_dim or args.key_dim
    num_heads = args.layer_heads or args.heads
    expand_v = args.layer_expand_v
    if expand_v is None:
        expand_v = args.value_dim / args.key_dim
    if expand_v <= 0 or not float(expand_v).is_integer():
        raise ValueError('layer-expand-v must be a positive integer for this benchmark.')
    return hidden_size, head_dim, num_heads, float(expand_v)


def _make_layer_bundle(args: argparse.Namespace, dtype: torch.dtype, case: Case, seed: int) -> ComponentBundle:
    hidden_size, head_dim, num_heads, expand_v = _layer_dimensions(args)
    torch.manual_seed(seed)
    layer = GatedDeltaNet(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        expand_v=expand_v,
        mode='chunk',
        use_short_conv=True,
        conv_size=4,
        conv_bias=True,
    ).to(device='cuda', dtype=dtype).eval()
    generator = torch.Generator(device='cuda').manual_seed(seed + 10000)
    x = torch.randn(1, case.t_max, hidden_size, dtype=dtype, device='cuda', generator=generator)
    return ComponentBundle(layer, {'x': x})


def _clone_layer_bundle(args: argparse.Namespace, dtype: torch.dtype, source: ComponentBundle, length: int) -> ComponentBundle:
    source_layer = source.owner
    assert isinstance(source_layer, GatedDeltaNet)
    hidden_size, head_dim, num_heads, expand_v = _layer_dimensions(args)
    layer = GatedDeltaNet(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        expand_v=expand_v,
        mode='chunk',
        use_short_conv=True,
        conv_size=4,
        conv_bias=True,
    ).to(device='cuda', dtype=dtype).eval()
    layer.load_state_dict(source_layer.state_dict())
    return ComponentBundle(layer, {'x': source.tensors['x'][:, :length].clone()})


def _call_layer_bundle(
    args: argparse.Namespace,
    bundle: ComponentBundle,
    cu_seqlens: torch.Tensor,
    use_graph: bool,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    graph_t_max: int | None = None,
    graph_n_max: int | None = None,
    graph_nt_max: int | None = None,
) -> object:
    assert isinstance(bundle.owner, GatedDeltaNet)
    return bundle.owner(
        bundle.tensors['x'],
        cu_seqlens=cu_seqlens,
        use_graph=use_graph,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        graph_t_max=graph_t_max if use_graph else None,
        graph_n_max=graph_n_max if use_graph else None,
        graph_nt_max=graph_nt_max if use_graph else None,
        chunk_size=args.chunk_size,
    )


def _make_conv_bundle(args: argparse.Namespace, dtype: torch.dtype, case: Case, seed: int) -> ComponentBundle:
    generator = torch.Generator(device='cuda').manual_seed(seed)
    dim = args.conv_dim or args.value_dim
    width = args.conv_width
    tensors = {
        'x': torch.randn(1, case.t_max, dim, dtype=dtype, device='cuda', generator=generator),
        'weight': torch.randn(dim, width, dtype=dtype, device='cuda', generator=generator),
        'bias': torch.randn(dim, dtype=dtype, device='cuda', generator=generator),
        'residual': torch.randn(1, case.t_max, dim, dtype=dtype, device='cuda', generator=generator),
        'initial_state': torch.randn(case.n_max, dim, width, dtype=dtype, device='cuda', generator=generator),
    }
    return ComponentBundle(None, tensors)


def _clone_conv_bundle(source: ComponentBundle, length: int) -> ComponentBundle:
    tensors = {
        name: value[:, :length].clone() if name in ('x', 'residual') else value.clone()
        for name, value in source.tensors.items()
    }
    return ComponentBundle(None, tensors)


def _call_conv_bundle(
    args: argparse.Namespace,
    bundle: ComponentBundle,
    cu_seqlens: torch.Tensor,
    use_graph: bool,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    graph_t_max: int | None = None,
    graph_n_max: int | None = None,
    graph_nt_max: int | None = None,
) -> object:
    return causal_conv1d(
        x=bundle.tensors['x'],
        weight=bundle.tensors['weight'],
        bias=bundle.tensors['bias'],
        residual=bundle.tensors['residual'],
        initial_state=bundle.tensors['initial_state'],
        output_final_state=True,
        activation='silu',
        backend='triton',
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=args.chunk_size,
        use_graph=use_graph,
        graph_nt_max=graph_nt_max if use_graph else None,
    )


def _make_kda_bundle(args: argparse.Namespace, dtype: torch.dtype, case: Case, seed: int) -> ComponentBundle:
    generator = torch.Generator(device='cuda').manual_seed(seed)
    heads = args.kda_heads or args.heads
    value_heads = args.kda_value_heads or args.value_heads
    key_dim = args.kda_key_dim or args.key_dim
    value_dim = args.kda_value_dim or args.value_dim
    if value_heads % heads:
        raise ValueError('kda-value-heads must be divisible by kda-heads.')
    q = torch.randn(1, case.t_max, heads, key_dim, dtype=dtype, device='cuda', generator=generator) * 0.2
    k = torch.randn(1, case.t_max, heads, key_dim, dtype=dtype, device='cuda', generator=generator) * 0.2
    v = torch.randn(1, case.t_max, value_heads, value_dim, dtype=dtype, device='cuda', generator=generator) * 0.1
    g = torch.randn(1, case.t_max, value_heads, key_dim, dtype=torch.float32, device='cuda', generator=generator) * 0.1
    beta = torch.randn(1, case.t_max, value_heads, dtype=dtype, device='cuda', generator=generator) * 0.2
    tensors = {
        'q': q,
        'k': k,
        'v': v,
        'g': g,
        'beta': beta,
        'A_log': torch.full((value_heads,), -0.5, dtype=torch.float32, device='cuda'),
        'dt_bias': torch.zeros(value_heads * key_dim, dtype=torch.float32, device='cuda'),
        'initial_state': torch.randn(
            case.n_max, value_heads, value_dim, key_dim, dtype=torch.float32, device='cuda', generator=generator
        ) * 0.01,
    }
    return ComponentBundle(None, tensors)


def _clone_kda_bundle(source: ComponentBundle, length: int) -> ComponentBundle:
    token_names = {'q', 'k', 'v', 'g', 'beta'}
    tensors = {
        name: value[:, :length].clone() if name in token_names else value.clone()
        for name, value in source.tensors.items()
    }
    return ComponentBundle(None, tensors)


def _call_kda_bundle(
    args: argparse.Namespace,
    bundle: ComponentBundle,
    cu_seqlens: torch.Tensor,
    use_graph: bool,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    graph_t_max: int | None = None,
    graph_n_max: int | None = None,
    graph_nt_max: int | None = None,
) -> object:
    return chunk_kda(
        q=bundle.tensors['q'],
        k=bundle.tensors['k'],
        v=bundle.tensors['v'],
        g=bundle.tensors['g'],
        beta=bundle.tensors['beta'],
        A_log=bundle.tensors['A_log'],
        dt_bias=bundle.tensors['dt_bias'],
        initial_state=bundle.tensors['initial_state'],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        chunk_size=args.chunk_size,
        use_graph=use_graph,
        graph_t_max=graph_t_max if use_graph else None,
        graph_n_max=graph_n_max if use_graph else None,
        graph_nt_max=graph_nt_max if use_graph else None,
    )


def _kda_layer_dimensions(args: argparse.Namespace) -> tuple[int, int, int, int, float]:
    num_heads = args.kda_heads or args.heads
    num_v_heads = args.kda_value_heads or args.value_heads
    head_dim = args.kda_key_dim or args.key_dim
    value_dim = args.kda_value_dim or args.value_dim
    hidden_size = args.layer_hidden_size or num_heads * head_dim
    expand_v = args.layer_expand_v if args.layer_expand_v is not None else value_dim / head_dim
    if expand_v <= 0 or not float(expand_v).is_integer():
        raise ValueError('KDA layer expand-v must be a positive integer for this benchmark.')
    return hidden_size, head_dim, num_heads, num_v_heads, float(expand_v)


def _make_kda_layer_bundle(args: argparse.Namespace, dtype: torch.dtype, case: Case, seed: int) -> ComponentBundle:
    hidden_size, head_dim, num_heads, num_v_heads, expand_v = _kda_layer_dimensions(args)
    torch.manual_seed(seed)
    layer = KimiDeltaAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        expand_v=expand_v,
        mode='chunk',
        use_short_conv=True,
        conv_size=4,
        conv_bias=True,
    ).to(device='cuda', dtype=dtype).eval()
    generator = torch.Generator(device='cuda').manual_seed(seed + 10000)
    x = torch.randn(1, case.t_max, hidden_size, dtype=dtype, device='cuda', generator=generator)
    return ComponentBundle(layer, {'x': x})


def _clone_kda_layer_bundle(
    args: argparse.Namespace,
    dtype: torch.dtype,
    source: ComponentBundle,
    length: int,
) -> ComponentBundle:
    source_layer = source.owner
    assert isinstance(source_layer, KimiDeltaAttention)
    hidden_size, head_dim, num_heads, num_v_heads, expand_v = _kda_layer_dimensions(args)
    layer = KimiDeltaAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        expand_v=expand_v,
        mode='chunk',
        use_short_conv=True,
        conv_size=4,
        conv_bias=True,
    ).to(device='cuda', dtype=dtype).eval()
    layer.load_state_dict(source_layer.state_dict())
    return ComponentBundle(layer, {'x': source.tensors['x'][:, :length].clone()})


def _call_kda_layer_bundle(
    args: argparse.Namespace,
    bundle: ComponentBundle,
    cu_seqlens: torch.Tensor,
    use_graph: bool,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    graph_t_max: int | None = None,
    graph_n_max: int | None = None,
    graph_nt_max: int | None = None,
) -> object:
    assert isinstance(bundle.owner, KimiDeltaAttention)
    return bundle.owner(
        bundle.tensors['x'],
        cu_seqlens=cu_seqlens,
        use_graph=use_graph,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        graph_t_max=graph_t_max if use_graph else None,
        graph_n_max=graph_n_max if use_graph else None,
        graph_nt_max=graph_nt_max if use_graph else None,
        chunk_size=args.chunk_size,
    )


def _run_mode(
    args: argparse.Namespace,
    capacity_case: Case,
    cases: list[Case],
    dtype: torch.dtype,
    mode: str,
) -> list[dict[str, object]]:
    requires_grad = mode == 'fwdbwd'
    context = torch.enable_grad() if requires_grad else torch.inference_mode()
    rows = []
    with context:
        torch.cuda.reset_peak_memory_stats()
        static_inputs = _make_inputs(
            capacity_case,
            args.heads,
            args.value_heads,
            args.key_dim,
            args.value_dim,
            dtype,
            args.seed,
            requires_grad,
        )
        static_cu = torch.tensor(capacity_case.cu_seqlens, dtype=torch.long, device='cuda')
        static_indices, static_offsets = _static_metadata(
            static_cu,
            args.chunk_size,
            capacity_case.nt_max,
        )
        static_do, static_dht = (None, None)
        if requires_grad:
            static_do, static_dht = _make_output_grads(
                capacity_case,
                args.value_heads,
                args.key_dim,
                args.value_dim,
                dtype,
                args.seed + 1000,
            )

        if requires_grad:
            graph_step = partial(
                _backward_step,
                static_inputs,
                static_cu,
                static_do,
                static_dht,
                args.chunk_size,
                True,
                static_indices,
                static_offsets,
                capacity_case.t_max,
                capacity_case.n_max,
                capacity_case.nt_max,
            )
        else:
            graph_step = partial(
                _call,
                static_inputs,
                static_cu,
                args.chunk_size,
                True,
                static_indices,
                static_offsets,
                capacity_case.t_max,
                capacity_case.n_max,
                capacity_case.nt_max,
            )

        graph_warmup_ms = _warmup(graph_step, args.warmup)
        graph, graph_outputs, capture_ms = _capture(graph_step)
        replay_warmup_ms = _warmup(graph.replay, args.warmup)

        for case_index, case in enumerate(cases):
            source_inputs = _make_inputs(
                case,
                args.heads,
                args.value_heads,
                args.key_dim,
                args.value_dim,
                dtype,
                args.seed + case_index + 1,
                False,
            )
            source_cu = torch.tensor(case.cu_seqlens, dtype=torch.long, device='cuda')
            source_indices, source_offsets = _static_metadata(source_cu, args.chunk_size, case.nt_max)
            eager_fixed_inputs = _slice_inputs(source_inputs, case.t_max, requires_grad)
            eager_live_inputs = _slice_inputs(source_inputs, case.actual_t, requires_grad)
            eager_fixed_cu = source_cu.clone()
            eager_live_cu = source_cu.clone()

            source_do, source_dht = (None, None)
            eager_fixed_do, eager_fixed_dht = (None, None)
            eager_live_do, eager_live_dht = (None, None)
            if requires_grad:
                source_do, source_dht = _make_output_grads(
                    case,
                    args.value_heads,
                    args.key_dim,
                    args.value_dim,
                    dtype,
                    args.seed + case_index + 1001,
                )
                eager_fixed_do, eager_fixed_dht = source_do.clone(), source_dht.clone()
                eager_live_do, eager_live_dht = source_do[:, :case.actual_t].clone(), source_dht.clone()

            _copy_inputs(static_inputs, source_inputs)
            static_cu.copy_(source_cu)
            static_indices.copy_(source_indices)
            static_offsets.copy_(source_offsets)
            if requires_grad:
                static_do.copy_(source_do)
                static_dht.copy_(source_dht)

            # Validate the replayed metadata and padding before collecting timing samples.
            torch.cuda.synchronize()
            if requires_grad:
                _zero_grads(static_inputs)
                graph.replay()
                torch.cuda.synchronize()
                eager_smoke = _backward_step(
                    eager_live_inputs,
                    eager_live_cu,
                    eager_live_do,
                    eager_live_dht,
                    args.chunk_size,
                    False,
                )
            else:
                graph.replay()
                torch.cuda.synchronize()
                eager_smoke = _call(eager_live_inputs, eager_live_cu, args.chunk_size, False)
            torch.cuda.synchronize()
            _assert_operator_smoke(
                graph_outputs,
                eager_smoke,
                case.actual_t,
                static_inputs,
                eager_live_inputs,
                requires_grad,
            )
            metadata_ptrs = (static_cu.data_ptr(), static_indices.data_ptr(), static_offsets.data_ptr())

            if requires_grad:
                eager_fixed_step = partial(
                    _backward_step,
                    eager_fixed_inputs,
                    eager_fixed_cu,
                    eager_fixed_do,
                    eager_fixed_dht,
                    args.chunk_size,
                    False,
                )
                eager_live_step = partial(
                    _backward_step,
                    eager_live_inputs,
                    eager_live_cu,
                    eager_live_do,
                    eager_live_dht,
                    args.chunk_size,
                    False,
                )
            else:
                eager_fixed_step = partial(_call, eager_fixed_inputs, eager_fixed_cu, args.chunk_size, False)
                eager_live_step = partial(_call, eager_live_inputs, eager_live_cu, args.chunk_size, False)
            graph_update_step = partial(
                _graph_update_step,
                graph,
                static_inputs,
                source_inputs,
                static_cu,
                source_cu,
                static_do,
                source_do,
                static_dht,
                source_dht,
                static_indices,
                source_indices,
                static_offsets,
                source_offsets,
            )
            copy_step = partial(
                _copy_step,
                static_inputs,
                source_inputs,
                static_cu,
                source_cu,
                static_do,
                source_do,
                static_dht,
                source_dht,
                static_indices,
                source_indices,
                static_offsets,
                source_offsets,
            )

            torch.cuda.reset_peak_memory_stats()
            _warmup(eager_fixed_step, args.warmup)
            _warmup(eager_live_step, args.warmup)
            _warmup(graph_update_step, args.warmup)
            _warmup(copy_step, args.warmup)
            eager_live = _measure(eager_live_step, args.iterations, args.repeats)
            eager_fixed = _measure(eager_fixed_step, args.iterations, args.repeats)
            graph_replay = _measure(graph.replay, args.iterations, args.repeats)
            graph_update = _measure(graph_update_step, args.iterations, args.repeats)
            copy_only = _measure(copy_step, args.iterations, args.repeats)
            decision = route_graph_execution(
                'auto',
                actual_tokens=case.actual_t,
                actual_sequences=sum(length > 0 for length in _make_lengths(case.actual_t, case.n_max, case.layout)),
                actual_nt=case.actual_nt,
                t_max=case.t_max,
                n_max=case.n_max,
                nt_max=case.nt_max,
                min_graph_utilization=args.min_graph_utilization,
                chunk_size=args.chunk_size,
                input_tokens=case.t_max,
                input_sequences=case.n_max,
            )
            auto_timing = graph_update if decision.selected_path == 'graph' else eager_live
            peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            rows.append({
                'component': 'operator',
                'mode': mode,
                'layout': case.layout,
                'route': decision.selected_path,
                'route_reason': decision.reason,
                'T_max': case.t_max,
                'N_max': case.n_max,
                'NT_max': case.nt_max,
                'actual_t': case.actual_t,
                'actual_NT': case.actual_nt,
                'occupancy': case.actual_t / case.t_max,
                'nt_occupancy': case.actual_nt / case.nt_max,
                'utilization': decision.utilization,
                'actual_sequences': decision.actual_sequences,
                'eager_live_ms': eager_live.wall_ms,
                'eager_fixed_ms': eager_fixed.wall_ms,
                'graph_replay_ms': graph_replay.wall_ms,
                'graph_update_ms': graph_update.wall_ms,
                'copy_only_ms': copy_only.wall_ms,
                'auto_ms': auto_timing.wall_ms,
                'eager_live_gpu_ms': eager_live.gpu_ms,
                'eager_fixed_gpu_ms': eager_fixed.gpu_ms,
                'graph_replay_gpu_ms': graph_replay.gpu_ms,
                'graph_update_gpu_ms': graph_update.gpu_ms,
                'copy_only_gpu_ms': copy_only.gpu_ms,
                'eager_live_range_ms': [eager_live.wall_min_ms, eager_live.wall_max_ms],
                'eager_fixed_range_ms': [eager_fixed.wall_min_ms, eager_fixed.wall_max_ms],
                'graph_replay_range_ms': [graph_replay.wall_min_ms, graph_replay.wall_max_ms],
                'graph_update_range_ms': [graph_update.wall_min_ms, graph_update.wall_max_ms],
                'eager_live_p95_ms': eager_live.wall_p95_ms,
                'eager_fixed_p95_ms': eager_fixed.wall_p95_ms,
                'graph_replay_p95_ms': graph_replay.wall_p95_ms,
                'graph_update_p95_ms': graph_update.wall_p95_ms,
                'copy_only_p95_ms': copy_only.wall_p95_ms,
                'auto_p95_ms': auto_timing.wall_p95_ms,
                'speedup_live_update': _speedup(eager_live, graph_update),
                'speedup_fixed_update': _speedup(eager_fixed, graph_update),
                'speedup_fixed_replay': _speedup(eager_fixed, graph_replay),
                'speedup_auto_vs_eager_fixed': _speedup(eager_fixed, auto_timing),
                'capture_ms': capture_ms,
                'graph_warmup_ms': graph_warmup_ms,
                'replay_warmup_ms': replay_warmup_ms,
                'break_even_fixed_replays': _break_even(capture_ms, eager_fixed, graph_update),
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / (1024 ** 3),
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / (1024 ** 3),
                'current_allocated_gib': torch.cuda.memory_allocated() / (1024 ** 3),
                'current_reserved_gib': torch.cuda.memory_reserved() / (1024 ** 3),
                'metadata_ptr_stable': metadata_ptrs == (
                    static_cu.data_ptr(), static_indices.data_ptr(), static_offsets.data_ptr()
                ),
                'correctness_smoke': True,
                'auto_semantics': 'external_scheduler: graph_update (copy+replay) or eager_live',
            })
            print(
                f"{mode:<7} {case.layout:<8} {decision.selected_path:<6} {case.t_max:>6} {case.n_max:>5} {case.nt_max:>7} "
                f"{case.actual_t:>8} {case.actual_nt:>9} {case.actual_t / case.t_max:>6.2f} "
                f"{eager_live.wall_ms:>10.3f} {eager_fixed.wall_ms:>11.3f} "
                f"{graph_replay.wall_ms:>11.3f} {graph_update.wall_ms:>11.3f} "
                f"{_speedup(eager_live, graph_update):>8.2f}x {_speedup(eager_fixed, graph_update):>8.2f}x "
                f"{capture_ms:>9.1f} {str(_break_even(capture_ms, eager_fixed, graph_update) or '-'):>5} "
                f"{peak_gib:>7.2f}",
                flush=True,
            )
            del source_inputs, eager_fixed_inputs, eager_live_inputs, source_indices, source_offsets
            del source_cu, eager_fixed_cu, eager_live_cu, eager_smoke
            if requires_grad:
                del source_do, source_dht, eager_fixed_do, eager_fixed_dht, eager_live_do, eager_live_dht
            gc.collect()

        del graph_outputs, graph, static_inputs, static_cu
        if requires_grad:
            del static_do, static_dht
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t-max', type=int, nargs='+', default=[2048, 4096, 8192])
    parser.add_argument('--n-max', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--actual-t', type=int, nargs='+', default=None)
    parser.add_argument('--actual-ratio', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--layouts', nargs='+', choices=LAYOUTS, default=['balanced'])
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--value-heads', type=int, default=16)
    parser.add_argument('--key-dim', type=int, default=128)
    parser.add_argument('--value-dim', type=int, default=128)
    parser.add_argument('--chunk-size', type=int, choices=(16, 32, 64), default=64)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--modes', nargs='+', choices=('fwd', 'fwdbwd'), default=['fwd', 'fwdbwd'])
    parser.add_argument(
        '--components', nargs='+', choices=('operator', 'layer', 'conv', 'kda', 'kda_layer'), default=['operator'],
        help='Benchmark components; fwd and fwdbwd are supported for every component.',
    )
    parser.add_argument('--layer-hidden-size', type=int, default=None)
    parser.add_argument('--layer-head-dim', type=int, default=None)
    parser.add_argument('--layer-heads', type=int, default=None)
    parser.add_argument('--layer-expand-v', type=float, default=None)
    parser.add_argument('--conv-dim', type=int, default=None)
    parser.add_argument('--conv-width', type=int, default=4)
    parser.add_argument('--kda-heads', type=int, default=None)
    parser.add_argument('--kda-value-heads', type=int, default=None)
    parser.add_argument('--kda-key-dim', type=int, default=None)
    parser.add_argument('--kda-value-dim', type=int, default=None)
    parser.add_argument('--include-fallback', action='store_true',
                        help='Benchmark auto-route eager fallback for T_max/N_max overflow.')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-graph-utilization', type=float, default=0.75)
    parser.add_argument('--json', type=Path, default=None, help='Write raw timing rows to this JSON file.')
    return parser.parse_args()


def _cases_for_capacity(args: argparse.Namespace, t_max: int, n_max: int) -> list[Case]:
    if args.actual_t is None:
        actual_values = [max(1, round(t_max * ratio)) for ratio in args.actual_ratio]
    else:
        actual_values = args.actual_t
    unique_values = sorted({value for value in actual_values if 1 <= value <= t_max})
    if not unique_values:
        raise ValueError(f'No actual token length is valid for T_max={t_max}.')
    return [
        _make_case(t_max, n_max, actual_t, args.chunk_size, layout)
        for actual_t in unique_values
        for layout in args.layouts
    ]


def _make_shape_inputs(
    token_length: int,
    sequence_count: int,
    heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device='cuda').manual_seed(seed)
    return {
        'q': torch.randn(1, token_length, heads, key_dim, dtype=dtype, device='cuda', generator=generator) * 0.4,
        'k': torch.randn(1, token_length, heads, key_dim, dtype=dtype, device='cuda', generator=generator) * 0.4,
        'v': torch.randn(1, token_length, value_heads, value_dim, dtype=dtype, device='cuda', generator=generator) * 0.1,
        'g': torch.randn(1, token_length, value_heads, dtype=torch.float32, device='cuda', generator=generator) * 0.2,
        'beta': torch.randn(1, token_length, value_heads, dtype=dtype, device='cuda', generator=generator) * 0.4,
        'initial_state': torch.randn(
            sequence_count, value_heads, value_dim, key_dim, dtype=torch.float32, device='cuda', generator=generator
        ) * 0.03,
        'A_log': -0.7 + torch.rand(value_heads, dtype=torch.float32, device='cuda', generator=generator) * 0.2,
        'dt_bias': torch.randn(value_heads, dtype=torch.float32, device='cuda', generator=generator) * 0.1,
    }


def _call_operator_with_route(
    inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    args: argparse.Namespace,
    graph_mode: str | None,
    graph_t_max: int,
    graph_n_max: int,
    graph_nt_max: int,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs = dict(
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
        cu_seqlens_cpu=cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu(),
        chunk_size=args.chunk_size,
        graph_t_max=graph_t_max,
        graph_n_max=graph_n_max,
        graph_nt_max=graph_nt_max,
    )
    if graph_mode is not None:
        kwargs['graph_mode'] = graph_mode
    return chunk_gated_delta_rule(**kwargs)[0]


def _run_fallback_benchmarks(args: argparse.Namespace, dtype: torch.dtype) -> list[dict[str, object]]:
    """Measure auto-route eager fallback without allocating or capturing a graph."""
    capacity_t = args.t_max[0]
    capacity_n = args.n_max[0]
    capacity_nt = _ceil_div(capacity_t, args.chunk_size) + capacity_n - 1
    over_t = capacity_t + max(1, capacity_t // 4)
    over_n = capacity_n + max(1, capacity_n // 2)
    scenarios = []

    token_cu = (0, over_t)
    token_nt = _ceil_div(over_t, args.chunk_size)
    scenarios.append(('fallback_tokens_over_t_max', over_t, 1, token_cu, token_nt))

    lengths = [capacity_t // over_n] * over_n
    lengths[-1] += capacity_t - sum(lengths)
    sequence_cu = [0]
    for length in lengths:
        sequence_cu.append(sequence_cu[-1] + length)
    sequence_nt = sum(_ceil_div(length, args.chunk_size) for length in lengths if length)
    scenarios.append(('fallback_sequences_over_n_max', capacity_t, over_n, tuple(sequence_cu), sequence_nt))

    rows = []
    with torch.inference_mode():
        for index, (scenario, token_length, sequence_count, offsets, actual_nt) in enumerate(scenarios):
            inputs = _make_shape_inputs(
                token_length,
                sequence_count,
                args.heads,
                args.value_heads,
                args.key_dim,
                args.value_dim,
                dtype,
                args.seed + 5000 + index,
            )
            cu_seqlens = torch.tensor(offsets, dtype=torch.long, device='cuda')
            cu_seqlens_cpu = cu_seqlens.cpu()
            decision = route_graph_execution(
                'auto',
                actual_tokens=token_length,
                actual_sequences=sequence_count,
                actual_nt=actual_nt,
                t_max=capacity_t,
                n_max=capacity_n,
                nt_max=capacity_nt,
                min_graph_utilization=args.min_graph_utilization,
                chunk_size=args.chunk_size,
            )
            assert decision.selected_path == 'eager'
            auto_step = partial(
                _call_operator_with_route,
                inputs,
                cu_seqlens,
                args,
                'auto',
                capacity_t,
                capacity_n,
                capacity_nt,
                cu_seqlens_cpu,
            )
            eager_step = partial(
                _call_operator_with_route,
                inputs,
                cu_seqlens,
                args,
                None,
                capacity_t,
                capacity_n,
                capacity_nt,
                cu_seqlens_cpu,
            )
            _warmup(auto_step, args.warmup)
            _warmup(eager_step, args.warmup)
            auto_timing = _measure(auto_step, args.iterations, args.repeats)
            eager_timing = _measure(eager_step, args.iterations, args.repeats)
            auto_output = auto_step()
            eager_output = eager_step()
            torch.testing.assert_close(auto_output, eager_output, atol=0.02, rtol=0.02)
            max_abs = (auto_output.float() - eager_output.float()).abs().max().item()
            rows.append({
                'component': 'operator',
                'mode': 'fwd',
                'scenario': scenario,
                'route': decision.selected_path,
                'route_reason': decision.reason,
                'T_max': capacity_t,
                'N_max': capacity_n,
                'NT_max': capacity_nt,
                'actual_t': token_length,
                'actual_sequences': sequence_count,
                'actual_NT': actual_nt,
                'utilization': decision.utilization,
                'auto_ms': auto_timing.wall_ms,
                'auto_p95_ms': auto_timing.wall_p95_ms,
                'eager_ms': eager_timing.wall_ms,
                'eager_p95_ms': eager_timing.wall_p95_ms,
                'speedup_auto': _speedup(eager_timing, auto_timing),
                'capture_ms': None,
                'graph_replay_ms': None,
                'graph_update_ms': None,
                'copy_only_ms': None,
                'correctness_max_abs': max_abs,
                'graph_capture_attempted': False,
                'graph_buffers_created': False,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / (1024 ** 3),
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / (1024 ** 3),
                'current_allocated_gib': torch.cuda.memory_allocated() / (1024 ** 3),
                'current_reserved_gib': torch.cuda.memory_reserved() / (1024 ** 3),
            })
            print(
                f"{'operator':<10} {scenario:<32} eager route reason={decision.reason:<24} "
                f"auto={auto_timing.wall_ms:.3f}ms eager={eager_timing.wall_ms:.3f}ms",
                flush=True,
            )
            del inputs, cu_seqlens
            gc.collect()
    torch.cuda.empty_cache()
    return rows


def _component_dimensions(args: argparse.Namespace, component: str) -> dict[str, int | float | bool]:
    if component == 'operator':
        return {'heads': args.heads, 'value_heads': args.value_heads, 'key_dim': args.key_dim, 'value_dim': args.value_dim}
    if component == 'layer':
        hidden_size, head_dim, num_heads, expand_v = _layer_dimensions(args)
        return {
            'hidden_size': hidden_size,
            'head_dim': head_dim,
            'num_heads': num_heads,
            'expand_v': expand_v,
            'use_short_conv': True,
        }
    if component == 'conv':
        return {'conv_dim': args.conv_dim or args.value_dim, 'conv_width': args.conv_width}
    if component == 'kda':
        return {
            'heads': args.kda_heads or args.heads,
            'value_heads': args.kda_value_heads or args.value_heads,
            'key_dim': args.kda_key_dim or args.key_dim,
            'value_dim': args.kda_value_dim or args.value_dim,
        }
    if component == 'kda_layer':
        hidden_size, head_dim, num_heads, num_v_heads, expand_v = _kda_layer_dimensions(args)
        return {
            'hidden_size': hidden_size,
            'head_dim': head_dim,
            'num_heads': num_heads,
            'num_v_heads': num_v_heads,
            'expand_v': expand_v,
            'use_short_conv': True,
        }
    raise ValueError(f'Unsupported component {component!r}.')


def _attach_run_metadata(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    common = {
        'dtype': args.dtype,
        'chunk_size': args.chunk_size,
        'warmup_iterations': args.warmup,
        'timed_iterations': args.iterations,
        'repeats': args.repeats,
        'seed': args.seed,
        'min_graph_utilization': args.min_graph_utilization,
        'git_ref': _git_label(),
        'device_name': torch.cuda.get_device_name(),
        'visible_cuda_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'python_version': platform.python_version(),
        'pytorch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'triton_version': triton.__version__,
    }
    for row in rows:
        row.setdefault('graph_capture_attempted', row.get('capture_ms') is not None)
        row.setdefault('graph_buffers_created', row.get('capture_ms') is not None)
        row.update(common)
        row.update(_component_dimensions(args, str(row['component'])))


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires an NVIDIA CUDA device.')
    if min(args.t_max + args.n_max + [args.heads, args.value_heads, args.key_dim, args.value_dim,
                                      args.warmup, args.iterations, args.repeats]) < 1:
        raise ValueError('All dimensions, warmup, iterations, and repeats must be positive.')
    if any(n_max < 1 for n_max in args.n_max):
        raise ValueError('N_max must be positive.')
    if any(ratio <= 0 or ratio > 1 for ratio in args.actual_ratio):
        raise ValueError('Every actual ratio must be in (0, 1].')
    if not 0 <= args.min_graph_utilization <= 1:
        raise ValueError('min-graph-utilization must be in [0, 1].')
    if args.value_heads % args.heads != 0:
        raise ValueError('value-heads must be divisible by heads for GVA.')
    if args.conv_width < 1:
        raise ValueError('conv-width must be positive.')
    if args.kda_heads is not None and args.kda_heads < 1:
        raise ValueError('kda-heads must be positive.')
    if args.kda_value_heads is not None and args.kda_value_heads < 1:
        raise ValueError('kda-value-heads must be positive.')
    if any(component in args.components for component in ('kda', 'kda_layer')) and args.chunk_size not in (32, 64):
        raise ValueError('KDA benchmarks require --chunk-size 32 or 64.')

    dtype = _dtype(args.dtype)
    print('=' * 150)
    print(f'Machine: {torch.cuda.get_device_name()} | CUDA {torch.version.cuda} | PyTorch {torch.__version__}')
    print(f'Ref: {_git_label()} | H={args.heads} | HV={args.value_heads} | K={args.key_dim} | V={args.value_dim} | '
          f'BT={args.chunk_size} | dtype={args.dtype}')
    print(f'Warmup={args.warmup} | iterations={args.iterations} | repeats={args.repeats}')
    print(f'Components={", ".join(args.components)} | min_graph_utilization={args.min_graph_utilization:.2f}')
    print(
        'Primary metric: synchronized wall time; graph_update includes GPU copies of all inputs and metadata, '
        'plus dO/dHT for fwdbwd.'
    )
    print('speedup columns: live/update, fixed/update, and fixed/replay. Capture and JIT are excluded from per-call times.')
    rows = []
    if 'operator' in args.components:
        print('=' * 150)
        print(
            f"{'mode':<7} {'layout':<8} {'route':<6} {'T_max':>6} {'N_max':>5} {'NT_max':>7} {'actual_t':>8} {'actual_NT':>9} "
            f"{'occ':>6} {'eager_live':>10} {'eager_fixed':>11} {'graph_replay':>11} {'graph_update':>11} "
            f"{'live/up':>8} {'fixed/up':>9} {'capture':>9} {'BE':>5} {'peakGiB':>7}"
        )
        print('-' * 150)
        for t_max in args.t_max:
            for n_max in args.n_max:
                cases = _cases_for_capacity(args, t_max, n_max)
                capacity_case = cases[0]
                print(
                    f'Preparing operator T_max={t_max}, N_max={n_max}, NT_max={capacity_case.nt_max} '
                    f'({len(cases)} layouts)...',
                    flush=True,
                )
                for mode in args.modes:
                    rows.extend(_run_mode(args, capacity_case, cases, dtype, mode))

    component_names = [name for name in args.components if name != 'operator']
    if component_names:
        print('=' * 150)
        print(
            f"{'component':<10} {'layout':<8} {'route':<6} {'T_max':>6} {'N_max':>5} {'NT_max':>7} "
            f"{'actual_t':>8} {'actual_NT':>9} {'util':>6} {'eager':>10} {'replay':>10} "
            f"{'update':>10} {'copy':>9} {'auto':>10} {'capture':>9} {'BE':>5} {'peakGiB':>7}"
        )
        print('-' * 150)
        for t_max in args.t_max:
            for n_max in args.n_max:
                cases = _cases_for_capacity(args, t_max, n_max)
                print(
                    f'Preparing {", ".join(component_names)} T_max={t_max}, N_max={n_max}...',
                    flush=True,
                )
                for component in component_names:
                    if component == 'layer':
                        make_bundle = partial(_make_layer_bundle, args, dtype)
                        clone_bundle = partial(_clone_layer_bundle, args, dtype)
                        call_bundle = partial(_call_layer_bundle, args)
                    elif component == 'conv':
                        make_bundle = partial(_make_conv_bundle, args, dtype)
                        clone_bundle = _clone_conv_bundle
                        call_bundle = partial(_call_conv_bundle, args)
                    elif component == 'kda':
                        make_bundle = partial(_make_kda_bundle, args, dtype)
                        clone_bundle = _clone_kda_bundle
                        call_bundle = partial(_call_kda_bundle, args)
                    elif component == 'kda_layer':
                        make_bundle = partial(_make_kda_layer_bundle, args, dtype)
                        clone_bundle = partial(_clone_kda_layer_bundle, args, dtype)
                        call_bundle = partial(_call_kda_layer_bundle, args)
                    else:
                        raise ValueError(f'Unsupported component {component!r}.')
                    for mode in args.modes:
                        if mode == 'fwd':
                            rows.extend(_run_component_benchmark(
                                args,
                                component,
                                cases,
                                make_bundle,
                                clone_bundle,
                                call_bundle,
                            ))
                        else:
                            rows.extend(_run_component_backward_benchmark(
                                args,
                                component,
                                cases,
                                make_bundle,
                                clone_bundle,
                                call_bundle,
                            ))

    if args.include_fallback:
        print('=' * 150)
        print('Auto-route fallback measurements (no graph capture is attempted for overflow cases).')
        rows.extend(_run_fallback_benchmarks(args, dtype))

    _attach_run_metadata(rows, args)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + '\n')
        print(f'Raw rows written to {args.json}')
    print('=' * 150)
    print('RTX 4090 is a consumer reference device; these numbers are not an H100/H20 MR performance claim.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
