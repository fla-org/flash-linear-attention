# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Helpers shared by fixed-capacity platform-graph execution paths."""

import weakref
from dataclasses import dataclass
from math import ceil
from typing import Literal

import torch

_BUFFERS: dict[tuple, torch.Tensor] = {}
_OWNED_BUFFERS: dict[int, tuple[weakref.ReferenceType, dict[tuple, torch.Tensor]]] = {}


def _owner_finalizer(owner_id: int):
    def cleanup(owner_ref: weakref.ReferenceType) -> None:
        entry = _OWNED_BUFFERS.get(owner_id)
        if entry is not None and entry[0] is owner_ref:
            _OWNED_BUFFERS.pop(owner_id, None)

    return cleanup


GraphMode = Literal['eager', 'force_graph', 'auto']
GraphPath = Literal['eager', 'graph']


def is_graph_capable_device(device: torch.device | str) -> bool:
    """Return whether the device has a supported platform graph API."""
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device).split(':', 1)[0]
    return device_type in ('cuda', 'npu')


@dataclass(frozen=True)
class GraphRouteDecision:
    """Result of routing one request before graph capture or replay."""

    selected_path: GraphPath
    reason: str
    actual_tokens: int | None
    actual_sequences: int | None
    actual_nt: int | None
    t_max: int
    n_max: int
    nt_max: int
    utilization: float | None


def get_static_buffer(
    name: str,
    shape: tuple,
    dtype: torch.dtype,
    device: torch.device | str,
    *,
    owner: object | None = None,
) -> torch.Tensor:
    """Return a persistent buffer owned by one graph input set.

    ``owner`` is intentionally identity-based (for example, the graph's input
    tensor).  Graphs for different layers must not alias intermediates merely
    because their shapes happen to match.
    """
    key = (name, tuple(shape), dtype, str(device))
    if owner is None:
        buf = _BUFFERS.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=device)
            _BUFFERS[key] = buf
        return buf

    owner_id = id(owner)
    entry = _OWNED_BUFFERS.get(owner_id)
    if entry is None or entry[0]() is not owner:
        owner_ref = weakref.ref(owner, _owner_finalizer(owner_id))
        buffers: dict[tuple, torch.Tensor] = {}
        _OWNED_BUFFERS[owner_id] = (owner_ref, buffers)
    else:
        buffers = entry[1]
    buf = buffers.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        buffers[key] = buf
    return buf


def normalize_graph_mode(mode: str | None, use_graph: bool = False) -> GraphMode:
    """Normalize the graph strategy while preserving the legacy ``use_graph`` flag."""
    if mode is None:
        return 'force_graph' if use_graph else 'eager'
    if mode not in ('eager', 'force_graph', 'auto'):
        raise ValueError(f"graph_mode must be one of 'eager', 'force_graph', or 'auto', got {mode!r}")
    if use_graph and mode == 'eager':
        raise ValueError("use_graph=True conflicts with graph_mode='eager'")
    return mode


def static_chunk_capacity(t_max: int, n_max: int, chunk_size: int) -> int:
    """Return the maximum number of chunks for a token and sequence bucket."""
    if not isinstance(t_max, int) or t_max < 1:
        raise ValueError(f't_max must be a positive integer, got {t_max!r}')
    if not isinstance(n_max, int) or n_max < 1:
        raise ValueError(f'n_max must be a positive integer, got {n_max!r}')
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f'chunk_size must be a positive integer, got {chunk_size!r}')
    return ceil(t_max / chunk_size) + n_max - 1


def validate_graph_capacity(t_max: int, n_max: int, nt_max: int, chunk_size: int | None = None) -> None:
    """Validate the fixed shape capacities required by a graph contract."""
    if not all(isinstance(value, int) and value >= 1 for value in (t_max, n_max, nt_max)):
        raise ValueError(
            f'graph capacities must be positive integers, got '
            f't_max={t_max!r}, n_max={n_max!r}, nt_max={nt_max!r}'
        )
    if chunk_size is not None:
        required = static_chunk_capacity(t_max, n_max, chunk_size)
        if nt_max < required:
            raise ValueError(f'nt_max={nt_max} is smaller than the required capacity {required}')


def route_graph_execution(
    mode: str | None,
    *,
    use_graph: bool = False,
    actual_tokens: int | None,
    actual_sequences: int | None,
    actual_nt: int | None,
    t_max: int,
    n_max: int,
    nt_max: int,
    min_graph_utilization: float = 0.75,
    chunk_size: int | None = None,
    input_tokens: int | None = None,
    input_sequences: int | None = None,
) -> GraphRouteDecision:
    """Select eager or fixed-capacity graph execution without device synchronization.

    The actual counts must be host metadata or caller-provided hints. A missing count
    makes ``auto`` choose eager; ``force_graph`` remains capacity-based and validates
    the physical input shape at the operator boundary.
    """
    normalized = normalize_graph_mode(mode, use_graph)
    validate_graph_capacity(t_max, n_max, nt_max, chunk_size)
    if not isinstance(min_graph_utilization, (float, int)) or not 0.0 <= min_graph_utilization <= 1.0:
        raise ValueError(f'min_graph_utilization must be in [0, 1], got {min_graph_utilization!r}')

    utilization = None if actual_nt is None else actual_nt / nt_max
    if normalized == 'eager':
        return GraphRouteDecision(
            'eager', 'explicit_eager', actual_tokens, actual_sequences, actual_nt,
            t_max, n_max, nt_max, utilization,
        )

    for value, limit, label in (
        (actual_tokens, t_max, 'tokens'),
        (actual_sequences, n_max, 'sequences'),
        (actual_nt, nt_max, 'chunks'),
    ):
        if value is not None and value > limit:
            if normalized == 'force_graph':
                raise ValueError(f'force_graph capacity exceeded: actual_{label}={value} > {label}_max={limit}')
            reason = f'{label}_exceed_{"t_max" if label == "tokens" else "n_max" if label == "sequences" else "nt_max"}'
            return GraphRouteDecision(
                'eager', reason, actual_tokens, actual_sequences, actual_nt,
                t_max, n_max, nt_max, utilization,
            )

    if normalized == 'auto' and any(value is None for value in (actual_tokens, actual_sequences, actual_nt)):
        return GraphRouteDecision(
            'eager', 'missing_host_metadata', actual_tokens, actual_sequences, actual_nt,
            t_max, n_max, nt_max, utilization,
        )

    if input_tokens is not None and input_tokens != t_max:
        if normalized == 'force_graph':
            raise ValueError(f'force_graph input shape requires T_max={t_max}, got input_tokens={input_tokens}')
        return GraphRouteDecision(
            'eager', 'input_token_shape_mismatch', actual_tokens, actual_sequences, actual_nt,
            t_max, n_max, nt_max, utilization,
        )
    if input_sequences is not None and input_sequences != n_max:
        if normalized == 'force_graph':
            raise ValueError(f'force_graph input shape requires N_max={n_max}, got input_sequences={input_sequences}')
        return GraphRouteDecision(
            'eager', 'input_sequence_shape_mismatch', actual_tokens, actual_sequences, actual_nt,
            t_max, n_max, nt_max, utilization,
        )

    if normalized == 'auto' and utilization is not None and utilization < min_graph_utilization:
        return GraphRouteDecision(
            'eager', 'low_chunk_utilization', actual_tokens, actual_sequences, actual_nt,
            t_max, n_max, nt_max, utilization,
        )
    return GraphRouteDecision(
        'graph', 'forced_graph' if normalized == 'force_graph' else 'high_chunk_utilization',
        actual_tokens, actual_sequences, actual_nt, t_max, n_max, nt_max, utilization,
    )


def host_chunk_statistics(cu_seqlens: object, chunk_size: int) -> tuple[int, int, int]:
    """Return token, non-empty sequence, and chunk counts from host metadata."""
    if isinstance(cu_seqlens, torch.Tensor) and cu_seqlens.device.type != 'cpu':
        raise ValueError('host_chunk_statistics requires CPU cumulative lengths')
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f'chunk_size must be a positive integer, got {chunk_size!r}')
    values = cu_seqlens.tolist() if hasattr(cu_seqlens, 'tolist') else list(cu_seqlens)
    if len(values) < 2:
        raise ValueError('cu_seqlens must contain at least a start and an end offset')
    values = [int(value) for value in values]
    if values[0] != 0:
        raise ValueError('cu_seqlens must start at zero')
    lengths = [eos - bos for bos, eos in zip(values[:-1], values[1:], strict=True)]
    if any(length < 0 for length in lengths):
        raise ValueError('cu_seqlens must be non-decreasing')
    return values[-1], sum(length > 0 for length in lengths), sum(ceil(length / chunk_size) for length in lengths if length > 0)


__all__ = [
    'GraphMode',
    'GraphPath',
    'GraphRouteDecision',
    'get_static_buffer',
    'host_chunk_statistics',
    'is_graph_capable_device',
    'normalize_graph_mode',
    'route_graph_execution',
    'static_chunk_capacity',
    'validate_graph_capacity',
]
