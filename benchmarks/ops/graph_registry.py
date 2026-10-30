# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Registry and operator adapters for graph capture benchmarks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class GraphBenchmarkMetadata:
    """Normalized workload fields serialized into graph benchmark results."""

    B: int
    T: int
    H: int
    D: int
    N: int
    actual_T: int
    HV: int
    DV: int
    chunk_size: int
    dtype: str
    sequence_profile: str
    fused_options: bool = False
    safe_gate: bool = False
    allow_neg_eigval: bool = False
    disable_recompute: bool = False

    def __post_init__(self):
        for name in ('B', 'T', 'H', 'D', 'N', 'actual_T', 'HV', 'DV', 'chunk_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.actual_T > self.T:
            raise ValueError('actual_T must satisfy 0 < actual_T <= T')
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError('dtype must be a nonempty string')
        if self.sequence_profile not in ('balanced', 'ragged', 'empty_tail'):
            raise ValueError('sequence_profile must be balanced, ragged, or empty_tail')
        for name in ('fused_options', 'safe_gate', 'allow_neg_eigval', 'disable_recompute'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')


@dataclass
class GraphBenchmarkCase:
    """One fully materialized eager/graph benchmark case.

    The runner owns the graph lifecycle and timing. Operator adapters own the
    arguments, calls, and numerical comparison because those details vary
    across operators.
    """

    eager_step: Callable
    graph_step: Callable
    capture_args: tuple[Any, ...]
    eager_args: tuple[Any, ...]
    graph_args: tuple[Any, ...]
    run_once: Callable[[Callable, tuple[Any, ...], bool], dict[str, Any]]
    assert_close: Callable[[dict[str, Any], dict[str, Any]], None]
    metadata: GraphBenchmarkMetadata

    def __post_init__(self):
        if not isinstance(self.metadata, GraphBenchmarkMetadata):
            raise TypeError('metadata must be GraphBenchmarkMetadata')


@dataclass(frozen=True)
class GraphOpConfig:
    name: str
    build_case: Callable[[dict[str, Any], str, str, int], GraphBenchmarkCase]
    default_shapes: dict[str, dict[str, Any]]
    supported_modes: tuple[str, ...] = ("fwdbwd",)
    category: str = ""


_REGISTRY: dict[str, GraphOpConfig] = {}


def register_graph_op(config: GraphOpConfig) -> None:
    _REGISTRY[config.name] = config


def get_graph_op(name: str) -> GraphOpConfig:
    if name not in _REGISTRY:
        raise KeyError(f"Graph op '{name}' is not registered. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_graph_ops() -> list[str]:
    return sorted(_REGISTRY)


def _as_bool(shape: dict[str, Any], name: str, default: bool = False) -> bool:
    value = shape.get(name, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _normalize_graph_shape(shape: dict[str, Any], default_dtype: str) -> dict[str, Any]:
    unknown = set(shape) - {field.name for field in fields(GraphBenchmarkMetadata)}
    if unknown:
        raise ValueError(f'Unknown graph benchmark fields: {sorted(unknown)}')
    missing = [name for name in ("B", "T", "H", "D") if name not in shape]
    if missing:
        raise ValueError(f"Graph shape is missing required dimensions: {missing}")

    normalized = dict(shape)
    for name in ("B", "T", "H", "D"):
        normalized[name] = int(normalized[name])
    normalized["N"] = int(normalized.get("N", normalized["B"]))
    normalized["actual_T"] = int(normalized.get("actual_T", normalized["T"]))
    normalized["HV"] = int(normalized.get("HV", normalized["H"]))
    normalized["DV"] = int(normalized.get("DV", normalized["D"]))
    normalized["chunk_size"] = int(normalized.get("chunk_size", 64))
    normalized["dtype"] = str(normalized.get("dtype", default_dtype))
    normalized["sequence_profile"] = str(normalized.get("sequence_profile", "balanced"))

    for name in ("fused_options", "safe_gate", "allow_neg_eigval", "disable_recompute"):
        normalized[name] = _as_bool(normalized, name)

    return asdict(GraphBenchmarkMetadata(**normalized))


def normalize_kda_shape(shape: dict[str, Any], default_dtype: str = "float16") -> dict[str, Any]:
    """Validate a KDA graph shape while preserving the common B/T/H/D axes."""
    normalized = _normalize_graph_shape(shape, default_dtype)
    if normalized["B"] != 1:
        raise ValueError("KDA graph benchmark requires B=1 for flattened variable-length inputs")
    if min(normalized[name] for name in ("T", "H", "D", "N", "HV", "DV")) <= 0:
        raise ValueError("T, H, D, N, HV, and DV must all be positive")
    if not 0 < normalized["actual_T"] <= normalized["T"]:
        raise ValueError("actual_T must satisfy 0 < actual_T <= T")
    if normalized["HV"] % normalized["H"]:
        raise ValueError("HV must be divisible by H")
    if normalized["chunk_size"] not in (32, 64):
        raise ValueError("chunk_size must be 32 or 64")
    if normalized["dtype"] not in ("float16", "bfloat16"):
        raise ValueError("dtype must be float16 or bfloat16")
    if normalized["sequence_profile"] not in ("balanced", "ragged", "empty_tail"):
        raise ValueError("sequence_profile must be balanced, ragged, or empty_tail")
    if (normalized["safe_gate"] or normalized["allow_neg_eigval"]) and not normalized["fused_options"]:
        raise ValueError("safe_gate and allow_neg_eigval require fused_options")
    return normalized


def make_cu_seqlens(actual_t: int, num_seqs: int, profile: str) -> list[int]:
    """Create deterministic replay offsets for a variable-length shape."""

    if profile == "balanced":
        return [i * actual_t // num_seqs for i in range(num_seqs)] + [actual_t]
    if profile == "empty_tail":
        return [0] + [actual_t] * num_seqs
    if profile == "ragged":
        # Give the first sequence roughly half the tokens, then distribute the
        # remainder. Duplicate offsets naturally represent empty sequences.
        first = (actual_t + 1) // 2
        if num_seqs == 1:
            return [0, actual_t]
        tail = [first + i * (actual_t - first) // (num_seqs - 1) for i in range(1, num_seqs)]
        return [0, first, *tail]
    raise ValueError(f"Unknown sequence profile: {profile}")


def _build_kda_case(shape: dict[str, Any], mode: str, device: str, seed: int) -> GraphBenchmarkCase:
    import torch
    import torch.nn.functional as F

    from fla.ops.kda import chunk_kda

    shape = normalize_kda_shape(shape)
    if mode not in ("fwd", "fwdbwd"):
        raise ValueError(f"Unsupported KDA graph benchmark mode: {mode}")

    B, T, H, K = (shape[name] for name in ("B", "T", "H", "D"))
    N, actual_t, HV, V = (shape[name] for name in ("N", "actual_T", "HV", "DV"))
    dtype = getattr(torch, shape["dtype"])
    requires_grad = mode == "fwdbwd"
    fused_options = shape["fused_options"]
    state_v_first = fused_options

    def make_inputs(input_seed: int) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator(device).manual_seed(input_seed)
        q = torch.randn(B, T, H, K, dtype=dtype, device=device, generator=generator)
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=device, generator=generator)
        if not fused_options:
            q = F.normalize(q.float(), dim=-1).to(dtype)
            k = F.normalize(k, dim=-1)
        k = k.to(dtype)
        v = torch.randn(B, T, HV, V, dtype=dtype, device=device, generator=generator)
        if fused_options:
            g = torch.randn(B, T, HV, K, dtype=dtype, device=device, generator=generator)
            beta = torch.randn(B, T, HV, dtype=dtype, device=device, generator=generator)
        else:
            g = F.logsigmoid(torch.randn(B, T, HV, K, dtype=torch.float32, device=device, generator=generator))
            beta = torch.rand(B, T, HV, dtype=dtype, device=device, generator=generator)
        state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
        h0 = torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator) * 0.01
        A_log = torch.randn(HV, dtype=torch.float32, device=device, generator=generator)
        dt_bias = torch.randn(HV * K, dtype=torch.float32, device=device, generator=generator)
        return tuple(tensor.requires_grad_(requires_grad) for tensor in (q, k, v, g, beta, h0, A_log, dt_bias))

    capture_inputs = make_inputs(seed)
    graph_inputs = make_inputs(seed + 1)
    eager_inputs = tuple(
        tensor[:, :actual_t].detach().clone().requires_grad_(requires_grad) if index < 5
        else tensor.detach().clone().requires_grad_(requires_grad)
        for index, tensor in enumerate(graph_inputs)
    )
    capture_offsets = make_cu_seqlens(T, N, "balanced")
    replay_offsets = make_cu_seqlens(actual_t, N, shape["sequence_profile"])
    capture_cu = torch.tensor(capture_offsets, dtype=torch.long, device=device)
    graph_cu = torch.tensor(replay_offsets, dtype=torch.long, device=device)
    eager_cu = graph_cu.clone()

    generator = torch.Generator(device).manual_seed(seed + 2)
    do = torch.randn(B, T, HV, V, dtype=dtype, device=device, generator=generator)
    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    dht = torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator)

    options = {
        "use_qk_l2norm_in_kernel": fused_options,
        "use_gate_in_kernel": fused_options,
        "use_beta_sigmoid_in_kernel": fused_options,
        "allow_neg_eigval": shape["allow_neg_eigval"],
        "safe_gate": shape["safe_gate"],
        "lower_bound": -5.0 if shape["safe_gate"] else None,
        "disable_recompute": shape["disable_recompute"],
        "state_v_first": state_v_first,
        "chunk_size": shape["chunk_size"],
    }

    def call(inputs: tuple[torch.Tensor, ...], cu_seqlens: torch.Tensor, use_graph: bool):
        q, k, v, g, beta, h0, A_log, dt_bias = inputs
        kwargs = dict(
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            A_log=A_log if fused_options else None,
            dt_bias=dt_bias if fused_options else None,
            **options,
        )
        if use_graph:
            kwargs.update(use_graph=True, max_num_seqs=N)
        return chunk_kda(q, k, v, g, beta, **kwargs)

    def eager_step(q, k, v, g, beta, h0, A_log, dt_bias, cu_seqlens):
        return call((q, k, v, g, beta, h0, A_log, dt_bias), cu_seqlens, use_graph=False)

    def graph_step(q, k, v, g, beta, h0, A_log, dt_bias, cu_seqlens):
        return call((q, k, v, g, beta, h0, A_log, dt_bias), cu_seqlens, use_graph=True)

    active_indices = list(range(6))
    if fused_options:
        active_indices.extend((6, 7))
    input_names = ("q", "k", "v", "g", "beta", "h0", "A_log", "dt_bias")

    def run_once(step: Callable, args: tuple[Any, ...], collect: bool) -> dict[str, torch.Tensor]:
        inputs = args[:-1]
        for tensor in inputs:
            if tensor.grad is not None:
                tensor.grad.zero_()
        o, ht = step(*args)
        if mode == "fwdbwd":
            ((o * do[:, :o.shape[1]]).sum() + (ht * dht).sum()).backward()
        if not collect:
            return {}
        result = {
            "o": o[:, :actual_t].detach(),
            "ht": ht.detach(),
        }
        if mode == "fwdbwd":
            for index in active_indices:
                grad = inputs[index].grad
                if grad is None:
                    raise AssertionError(f"Expected a gradient for {input_names[index]}")
                if index < 5:
                    grad = grad[:, :actual_t]
                result[f"d{input_names[index]}"] = grad.detach()
        return result

    def assert_close(actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> None:
        if actual.keys() != reference.keys():
            raise AssertionError(f"Result keys differ: graph={actual.keys()}, eager={reference.keys()}")
        for name in actual:
            if not torch.isfinite(reference[name]).all():
                count = int((~torch.isfinite(reference[name])).sum())
                raise AssertionError(f"Eager reference {name} contains {count} non-finite value(s)")
            if not torch.isfinite(actual[name]).all():
                count = int((~torch.isfinite(actual[name])).sum())
                raise AssertionError(f"Graph result {name} contains {count} non-finite value(s)")
            torch.testing.assert_close(
                actual[name],
                reference[name],
                rtol=3e-3,
                atol=3e-3,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    return GraphBenchmarkCase(
        eager_step=eager_step,
        graph_step=graph_step,
        capture_args=capture_inputs + (capture_cu,),
        eager_args=eager_inputs + (eager_cu,),
        graph_args=graph_inputs + (graph_cu,),
        run_once=run_once,
        assert_close=assert_close,
        metadata=GraphBenchmarkMetadata(**shape),
    )


KDA_GRAPH_SHAPES = {
    "baseline_fp16": {
        "B": 1, "T": 128, "H": 2, "D": 64, "N": 2,
        "dtype": "float16", "chunk_size": 64,
    },
    "fused_bf16": {
        "B": 1, "T": 128, "H": 2, "D": 64, "N": 2,
        "dtype": "bfloat16", "chunk_size": 64, "fused_options": True,
        "safe_gate": True, "allow_neg_eigval": True,
    },
    "chunk32_bf16": {
        "B": 1, "T": 128, "H": 2, "D": 64, "N": 2,
        "dtype": "bfloat16", "chunk_size": 32, "fused_options": True,
    },
    "longer_sequence_fp16": {
        "B": 1, "T": 512, "H": 2, "D": 64, "N": 2,
        "dtype": "float16", "chunk_size": 64,
    },
}


register_graph_op(GraphOpConfig(
    name="chunk_kda",
    build_case=_build_kda_case,
    default_shapes=KDA_GRAPH_SHAPES,
    supported_modes=("fwd", "fwdbwd"),
    category="gate_beta",
))


def normalize_gdn_shape(shape: dict[str, Any], default_dtype: str = "float16") -> dict[str, Any]:
    """Normalize workload metadata; the operator validates graph capabilities when building the case."""
    return _normalize_graph_shape(shape, default_dtype)


def _check_gdn_error(name: str, abs_error: float, rms_error: float, tolerance: float) -> None:
    # keep the existing RMS/absolute budgets, independently of FLA_CI_ENV.
    if not math.isfinite(abs_error) or not math.isfinite(rms_error):
        raise AssertionError(f"Non-finite GDN {name} error")
    if abs_error > 1e-6 and rms_error >= tolerance:
        raise AssertionError(f"GDN {name}: abs={abs_error}, RMS={rms_error}, tolerance={tolerance}")


def _build_gdn_case(shape: dict[str, Any], mode: str, device: str, seed: int) -> GraphBenchmarkCase:
    if mode not in ("fwd", "fwdbwd"):
        raise ValueError(f"Unsupported GDN graph benchmark mode: {mode}")

    from importlib import import_module

    import torch
    import torch.nn.functional as F

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.utils.testing import get_abs_err, get_err_ratio

    # eager baselines may predate the graph module or its shared config validator.
    module_name = 'fla.ops.gated_delta_rule.graph'
    try:
        graph_module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        graph_module = None
    validate_config = getattr(graph_module, '_validate_graph_config', None)
    shape = normalize_gdn_shape(shape)
    if torch.device(device).type != "npu":
        raise ValueError("GDN graph benchmark requires NPU")
    options = {
        'chunk_size': shape['chunk_size'],
        'use_qk_l2norm_in_kernel': shape['fused_options'],
        'use_beta_sigmoid_in_kernel': shape['fused_options'],
        'use_gate_in_kernel': shape['fused_options'],
        'allow_neg_eigval': shape['allow_neg_eigval'],
    }
    for name in ('safe_gate', 'disable_recompute'):
        if shape[name]:
            options[name] = True
    if validate_config is not None:
        validate_config(
            q_shape=(shape['B'], shape['T'], shape['H'], shape['D']),
            v_shape=(shape['B'], shape['T'], shape['HV'], shape['DV']),
            dtype=shape['dtype'],
            options=options,
        )
    B, T, H, D = (shape[name] for name in ('B', 'T', 'H', 'D'))
    N, actual_t, HV, V = (shape[name] for name in ('N', 'actual_T', 'HV', 'DV'))
    dtype = getattr(torch, shape["dtype"])
    requires_grad = mode == "fwdbwd"

    def make_inputs(input_seed):
        gen = torch.Generator(device).manual_seed(input_seed)
        size = (B, T, H, D)
        q, k = [F.normalize(torch.randn(size, device=device, generator=gen), dim=-1).to(dtype) for _ in range(2)]
        v = torch.randn((B, T, HV, V), dtype=dtype, device=device, generator=gen)
        g = F.logsigmoid(torch.randn((B, T, HV), device=device, generator=gen)) / 8
        beta = torch.rand((B, T, HV), dtype=dtype, device=device, generator=gen)
        h0 = torch.randn(N, HV, D, V, device=device, generator=gen)
        return tuple(x.requires_grad_(requires_grad) for x in (q, k, v, g, beta, h0))

    capture_cu = torch.tensor(make_cu_seqlens(T, N, "balanced"), device=device)
    live_cu = torch.tensor(make_cu_seqlens(actual_t, N, shape["sequence_profile"]), device=device)
    capture_args = make_inputs(seed) + (capture_cu,)
    graph_args = make_inputs(seed + 1) + (live_cu,)
    eager_inputs = tuple(
        (x[:, :actual_t] if i < 5 else x).detach().clone().requires_grad_(requires_grad)
        for i, x in enumerate(graph_args[:-1])
    )
    eager_args = eager_inputs + (live_cu.clone(),)
    generator = torch.Generator(device).manual_seed(seed + 2)
    do = torch.randn((B, T, HV, V), dtype=dtype, device=device, generator=generator)
    dht = torch.randn((N, HV, D, V), device=device, generator=generator)
    input_names = ("q", "k", "v", "g", "beta", "h0")

    def call(q, k, v, g, beta, h0, cu, use_graph):
        return chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            use_graph=use_graph,
            **options,
        )

    def eager_step(*args):
        return call(*args, use_graph=False)

    def graph_step(*args):
        if graph_module is None:
            raise RuntimeError('This revision does not support GDN graph replay')
        return call(*args, use_graph=True)

    def run_once(step, args, collect):
        o, ht = step(*args)
        grads = None
        if requires_grad:
            grads = torch.autograd.grad((o, ht), args[:-1], (do[:, :o.shape[1]].contiguous(), dht))
        if not collect:
            return {}
        result = {"o": o[:, :actual_t].detach().clone(), "ht": ht.detach().clone()}
        if grads is not None:
            for index, (name, grad) in enumerate(zip(input_names, grads)):
                if index < 5:
                    if torch.count_nonzero(grad[:, actual_t:]):
                        raise AssertionError(f"Nonzero GDN inactive d{name}")
                    grad = grad[:, :actual_t]
                result[f"d{name}"] = grad.detach().clone()
        return result

    def compare(actual, reference):
        if actual.keys() != reference.keys():
            raise AssertionError(f"GDN result keys differ: {actual.keys()} != {reference.keys()}")
        tolerances = {"o": 0.005, "ht": 0.005, "dq": 0.007, "dk": 0.008,
                      "dv": 0.007, "dg": 0.015, "dbeta": 0.015, "dh0": 0.007}
        for name in reference:
            if not torch.isfinite(actual[name]).all() or not torch.isfinite(reference[name]).all():
                raise AssertionError(f"Non-finite GDN {name}")
            _check_gdn_error(
                name,
                get_abs_err(reference[name], actual[name]),
                get_err_ratio(reference[name], actual[name]),
                tolerances[name] if requires_grad else 3e-3,
            )

    return GraphBenchmarkCase(
        eager_step=eager_step,
        graph_step=graph_step,
        capture_args=capture_args,
        eager_args=eager_args,
        graph_args=graph_args,
        run_once=run_once,
        assert_close=compare,
        metadata=GraphBenchmarkMetadata(**shape),
    )


register_graph_op(GraphOpConfig(
    name="chunk_gdn",
    build_case=_build_gdn_case,
    default_shapes={
        f'{label}_{suffix}': {**KDA_GRAPH_SHAPES[source], 'dtype': dtype}
        for label, source in (('baseline', 'baseline_fp16'), ('longer_sequence', 'longer_sequence_fp16'))
        for suffix, dtype in (('fp16', 'float16'), ('bf16', 'bfloat16'))
    },
    supported_modes=("fwd", "fwdbwd"),
    category="gate_beta",
))
