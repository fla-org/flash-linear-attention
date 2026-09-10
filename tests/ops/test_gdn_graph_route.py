# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.utils.graph import static_chunk_capacity
from fla.ops.utils.index import prepare_chunk_indices_static
from fla.utils import IS_NVIDIA, device

T_MAX = 256
H = 1
HV = 1
K = 16
V = 16
CHUNK_SIZE = 64


@pytest.fixture
def route_spy(monkeypatch: pytest.MonkeyPatch):
    implementation = importlib.import_module('fla.ops.gated_delta_rule.chunk')
    original = implementation.chunk_gated_delta_rule_fwd
    calls = []

    def wrapped(*args, **kwargs):
        indices = kwargs.get('chunk_indices')
        calls.append((kwargs.get('use_graph', False), None if indices is None else tuple(indices.shape)))
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, 'chunk_gated_delta_rule_fwd', wrapped)
    return calls


def _make_inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(1, T_MAX, H, K, dtype=torch.float16, device=device, generator=generator)
    k = F.normalize(torch.randn_like(q).float(), dim=-1).to(q.dtype)
    v = torch.randn(1, T_MAX, HV, V, dtype=q.dtype, device=device, generator=generator)
    g = -torch.rand(1, T_MAX, HV, dtype=torch.float32, device=device, generator=generator) * 0.2
    beta = torch.rand(1, T_MAX, HV, dtype=q.dtype, device=device, generator=generator) * 0.8 + 0.1
    return {'q': q, 'k': k, 'v': v, 'g': g, 'beta': beta}


def _run(
    inputs: dict[str, torch.Tensor],
    offsets: tuple[int, ...],
    *,
    t_max: int,
    n_max: int,
    graph_mode: str | None,
):
    cu_seqlens = torch.tensor(offsets, dtype=torch.long, device=device)
    kwargs = dict(
        q=inputs['q'],
        k=inputs['k'],
        v=inputs['v'],
        g=inputs['g'],
        beta=inputs['beta'],
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens.cpu(),
        chunk_size=CHUNK_SIZE,
        graph_t_max=t_max,
        graph_n_max=n_max,
        graph_nt_max=static_chunk_capacity(t_max, n_max, CHUNK_SIZE),
    )
    if graph_mode is not None:
        kwargs['graph_mode'] = graph_mode
    return chunk_gated_delta_rule(**kwargs)[0]


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph routing requires an NVIDIA CUDA device')
@pytest.mark.parametrize(
    ('name', 'offsets', 't_max', 'n_max', 'expected_graph'),
    [
        ('high-utilization', (0, 128, 256), 256, 2, True),
        ('low-utilization', (0, 1, 1), 256, 2, False),
        ('tokens-over-capacity', (0, 128, 256), 128, 2, False),
        ('sequences-over-capacity', (0, 64, 128, 192, 256), 256, 2, False),
    ],
)
@torch.inference_mode()
def test_gdn_auto_route_executes_selected_path(
    name: str,
    offsets: tuple[int, ...],
    t_max: int,
    n_max: int,
    expected_graph: bool,
    route_spy,
):
    inputs = _make_inputs(seed=11)
    routed = _run(inputs, offsets, t_max=t_max, n_max=n_max, graph_mode='auto')
    assert route_spy, f'{name} did not execute the GDN implementation'
    use_graph, indices_shape = route_spy[-1]
    assert use_graph is expected_graph
    if expected_graph:
        assert indices_shape == (static_chunk_capacity(t_max, n_max, CHUNK_SIZE), 2)
    else:
        assert indices_shape is None or indices_shape[1] == 2

    reference = _run(inputs, offsets, t_max=t_max, n_max=n_max, graph_mode=None)
    actual_t = offsets[-1]
    torch.testing.assert_close(routed[:, :actual_t], reference[:, :actual_t], atol=0.02, rtol=0.02)
    assert torch.isfinite(routed[:, :actual_t]).all()
    if expected_graph:
        assert torch.isfinite(routed).all()


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph routing requires an NVIDIA CUDA device')
@torch.inference_mode()
def test_gdn_force_graph_rejects_over_capacity():
    inputs = _make_inputs(seed=23)
    with pytest.raises(ValueError, match='actual_tokens'):
        _run(
            inputs,
            (0, 128, 256),
            t_max=128,
            n_max=2,
            graph_mode='force_graph',
        )


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph routing requires an NVIDIA CUDA device')
@torch.inference_mode()
def test_gdn_auto_eager_does_not_use_graph_only_metadata(route_spy):
    inputs = _make_inputs(seed=31)
    cu_seqlens = torch.tensor((0, 1, 1), dtype=torch.long, device=device)
    nt_max = static_chunk_capacity(T_MAX, 2, CHUNK_SIZE)
    static_indices, static_offsets = prepare_chunk_indices_static(cu_seqlens, CHUNK_SIZE, nt_max)
    routed = chunk_gated_delta_rule(
        **inputs,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens.cpu(),
        chunk_size=CHUNK_SIZE,
        graph_mode='auto',
        graph_t_max=T_MAX,
        graph_n_max=2,
        graph_nt_max=nt_max,
        chunk_indices=static_indices,
        chunk_offsets=static_offsets,
    )[0]
    assert torch.isfinite(routed[:, :1]).all()
    assert route_spy[-1][0] is False
    # Eager receives dynamically-sized metadata, not the graph bucket.
    assert route_spy[-1][1] != (nt_max, 2)


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph routing requires an NVIDIA CUDA device')
@pytest.mark.parametrize(
    ('offsets', 't_max', 'n_max'),
    [
        ((0, 1, 1), 256, 2),
        ((0, 128, 256), 128, 2),
        ((0, 64, 128, 192, 256), 256, 2),
    ],
    ids=['low-utilization', 'tokens-over-capacity', 'sequences-over-capacity'],
)
@torch.inference_mode()
def test_gdn_auto_fallback_never_builds_static_metadata(
    offsets: tuple[int, ...],
    t_max: int,
    n_max: int,
    monkeypatch: pytest.MonkeyPatch,
):
    implementation = importlib.import_module('fla.ops.gated_delta_rule.chunk')

    def unexpected_static_metadata(*args, **kwargs):
        pytest.fail('auto eager fallback must not build graph-only static metadata')

    monkeypatch.setattr(implementation, 'prepare_chunk_indices_static', unexpected_static_metadata)
    output = _run(_make_inputs(seed=41), offsets, t_max=t_max, n_max=n_max, graph_mode='auto')
    assert torch.isfinite(output[:, :offsets[-1]]).all()


@pytest.mark.parametrize('graph_kwargs', [{'use_graph': True}, {'graph_mode': 'force_graph'}, {'graph_mode': 'auto'}])
def test_flash_qla_verifier_rejects_graph_modes(graph_kwargs):
    from fla.ops.gated_delta_rule.backends.flash_qla import FlashQLABackend

    tensor = torch.empty(1, 64, 1, 128, dtype=torch.bfloat16)
    passed, reason = FlashQLABackend().chunk_gated_delta_rule_verifier(
        q=tensor,
        k=tensor,
        v=tensor,
        g=torch.empty(1, 64, 1),
        beta=torch.empty(1, 64, 1),
        **graph_kwargs,
    )
    assert not passed
    assert reason == 'FlashQLA does not support CUDA Graph mode'


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph routing requires an NVIDIA CUDA device')
@torch.inference_mode()
def test_gdn_force_graph_requires_external_static_metadata():
    inputs = _make_inputs(seed=37)
    with pytest.raises(ValueError, match='caller-provided fixed'):
        _run(
            inputs,
            (0, 128, 256),
            t_max=T_MAX,
            n_max=2,
            graph_mode='force_graph',
        )
