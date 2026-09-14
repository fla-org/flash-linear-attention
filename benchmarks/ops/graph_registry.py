# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Registry and operator adapters for graph capture benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


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
    metadata: dict[str, Any] = field(default_factory=dict)


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


def normalize_kda_shape(shape: dict[str, Any], default_dtype: str = "float16") -> dict[str, Any]:
    """Validate a KDA graph shape while preserving the common B/T/H/D axes."""

    missing = [name for name in ("B", "T", "H", "D") if name not in shape]
    if missing:
        raise ValueError(f"KDA graph shape is missing required dimensions: {missing}")

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


def make_capture_cu_seqlens(total_tokens: int, num_seqs: int) -> list[int]:
    """Create a valid layout that reaches the tight static chunk capacity."""

    num_nonempty_seqs = min(total_tokens, num_seqs)
    return [*range(num_nonempty_seqs), *([total_tokens] * (num_seqs - num_nonempty_seqs + 1))]


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
    capture_offsets = make_capture_cu_seqlens(T, N)
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
        metadata=shape,
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
