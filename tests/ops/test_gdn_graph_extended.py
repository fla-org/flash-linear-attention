# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.utils.graph import static_chunk_capacity
from fla.ops.utils.index import prepare_chunk_indices_static
from fla.utils import IS_NVIDIA, assert_close, device


class GraphCase(NamedTuple):
    name: str
    t_max: int
    n_max: int
    h: int
    hv: int
    k: int
    v: int
    chunk_size: int
    dtype: torch.dtype
    use_qk_l2norm_in_kernel: bool
    use_gate_in_kernel: bool
    use_beta_sigmoid_in_kernel: bool
    allow_neg_eigval: bool
    state_v_first: bool
    output_final_state: bool
    layouts: tuple[tuple[int, ...], ...]


GRAPH_CASES = (
    GraphCase(
        'gva-fp16-k32-v64-bt16',
        256,
        4,
        2,
        4,
        32,
        64,
        16,
        torch.float16,
        False,
        False,
        False,
        False,
        False,
        True,
        (
            (0, 1, 2, 3, 256),
            (0, 16, 17, 240, 240),
            (0, 7, 64, 64, 64),
            (0, 64, 128, 192, 192),
        ),
    ),
    GraphCase(
        'bf16-fused-gate-beta-sigmoid',
        256,
        4,
        2,
        2,
        64,
        32,
        32,
        torch.bfloat16,
        True,
        True,
        True,
        True,
        False,
        True,
        (
            (0, 1, 2, 3, 256),
            (0, 32, 33, 224, 224),
            (0, 7, 64, 64, 64),
            (0, 64, 128, 192, 192),
        ),
    ),
    GraphCase(
        'state-v-first-k128-v64',
        256,
        4,
        1,
        2,
        128,
        64,
        64,
        torch.float16,
        False,
        False,
        False,
        False,
        True,
        True,
        (
            (0, 1, 2, 3, 256),
            (0, 64, 65, 256, 256),
            (0, 7, 128, 128, 128),
            (0, 64, 128, 192, 192),
        ),
    ),
    GraphCase(
        'no-initial-state-fp16',
        128,
        4,
        2,
        2,
        32,
        32,
        32,
        torch.float16,
        True,
        False,
        False,
        False,
        False,
        False,
        (
            (0, 1, 2, 3, 128),
            (0, 32, 33, 120, 120),
            (0, 7, 64, 64, 64),
        ),
    ),
)

TOKEN_INPUTS = ('q', 'k', 'v', 'g', 'beta')


def _grad_names(case: GraphCase) -> tuple[str, ...]:
    names = TOKEN_INPUTS + (('initial_state',) if case.output_final_state else ())
    if case.use_gate_in_kernel:
        names += ('A_log', 'dt_bias')
    return names


def _make_inputs(case: GraphCase, seed: int, requires_grad: bool = False) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device).manual_seed(seed)
    q = torch.randn(1, case.t_max, case.h, case.k, dtype=case.dtype, device=device, generator=generator) * 0.4
    k = torch.randn(1, case.t_max, case.h, case.k, dtype=case.dtype, device=device, generator=generator) * 0.4
    if not case.use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), dim=-1).to(case.dtype)
        k = F.normalize(k.float(), dim=-1).to(case.dtype)

    v = torch.randn(1, case.t_max, case.hv, case.v, dtype=case.dtype, device=device, generator=generator) * 0.1
    if case.use_gate_in_kernel:
        g = torch.randn(1, case.t_max, case.hv, dtype=torch.float32, device=device, generator=generator) * 0.2
        A_log = -0.7 + torch.rand(case.hv, dtype=torch.float32, device=device, generator=generator) * 0.2
        dt_bias = torch.randn(case.hv, dtype=torch.float32, device=device, generator=generator) * 0.1
    else:
        g = -(0.1 + torch.rand(1, case.t_max, case.hv, dtype=torch.float32, device=device, generator=generator) * 0.5)
        A_log = None
        dt_bias = None

    if case.use_beta_sigmoid_in_kernel:
        beta = torch.randn(1, case.t_max, case.hv, dtype=case.dtype, device=device, generator=generator) * 0.4
    else:
        beta = 0.2 + torch.rand(1, case.t_max, case.hv, dtype=case.dtype, device=device, generator=generator) * 0.6

    if case.output_final_state:
        state_shape = (case.n_max, case.hv, case.v, case.k) if case.state_v_first else (case.n_max, case.hv, case.k, case.v)
        initial_state = torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator) * 0.03
    else:
        initial_state = None

    inputs = {
        'q': q,
        'k': k,
        'v': v,
        'g': g,
        'beta': beta,
        'initial_state': initial_state,
        'A_log': A_log,
        'dt_bias': dt_bias,
    }
    if requires_grad:
        for value in inputs.values():
            if value is not None:
                value.requires_grad_(True)
    return inputs


def _make_output_grads(case: GraphCase, seed: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator(device).manual_seed(seed)
    do = torch.randn(1, case.t_max, case.hv, case.v, dtype=case.dtype, device=device, generator=generator) * 0.2
    if case.output_final_state:
        state_shape = (case.n_max, case.hv, case.v, case.k) if case.state_v_first else (case.n_max, case.hv, case.k, case.v)
        dht = torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator) * 0.2
    else:
        dht = None
    return do, dht


def _call(
    inputs: dict[str, torch.Tensor | None],
    cu_seqlens: torch.Tensor,
    case: GraphCase,
    use_graph: bool,
    *,
    graph_mode: str | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    graph_nt_max: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return chunk_gated_delta_rule(
        q=inputs['q'],
        k=inputs['k'],
        v=inputs['v'],
        g=inputs['g'],
        beta=inputs['beta'],
        initial_state=inputs['initial_state'],
        output_final_state=case.output_final_state,
        use_qk_l2norm_in_kernel=case.use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=case.use_beta_sigmoid_in_kernel,
        allow_neg_eigval=case.allow_neg_eigval,
        state_v_first=case.state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=case.chunk_size,
        use_gate_in_kernel=case.use_gate_in_kernel,
        A_log=inputs['A_log'],
        dt_bias=inputs['dt_bias'],
        use_graph=use_graph,
        graph_mode=graph_mode,
        graph_t_max=case.t_max,
        graph_n_max=case.n_max,
        graph_nt_max=graph_nt_max,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )


def _backward(
    outputs: tuple[torch.Tensor, torch.Tensor | None],
    output_grads: tuple[torch.Tensor, torch.Tensor | None],
) -> None:
    output, final_state = outputs
    do, dht = output_grads
    if final_state is None:
        torch.autograd.backward(output, do)
    else:
        torch.autograd.backward((output, final_state), (do, dht))


def _copy_inputs(dst: dict[str, torch.Tensor | None], src: dict[str, torch.Tensor | None]) -> None:
    with torch.no_grad():
        for name, value in dst.items():
            if value is not None:
                value.copy_(src[name])


def _slice_inputs(
    inputs: dict[str, torch.Tensor | None],
    actual_t: int,
    requires_grad: bool,
) -> dict[str, torch.Tensor | None]:
    result = {}
    for name, value in inputs.items():
        if value is None:
            result[name] = None
        elif name in TOKEN_INPUTS:
            result[name] = value[:, :actual_t].detach().clone()
        else:
            result[name] = value.detach().clone()
    if requires_grad:
        for name in _grad_names_for_inputs(result):
            result[name].requires_grad_(True)
    return result


def _grad_names_for_inputs(inputs: dict[str, torch.Tensor | None]) -> tuple[str, ...]:
    names = tuple(name for name in TOKEN_INPUTS if inputs[name] is not None)
    if inputs['initial_state'] is not None:
        names += ('initial_state',)
    if inputs['A_log'] is not None:
        names += ('A_log', 'dt_bias')
    return names


def _assert_finite(name: str, value: torch.Tensor | None) -> None:
    if value is not None:
        assert torch.isfinite(value).all(), f'{name} contains non-finite values'


def _assert_graph_padding(
    graph_output: torch.Tensor,
    graph_grads: dict[str, torch.Tensor],
    actual_t: int,
    case: GraphCase,
) -> None:
    if actual_t >= case.t_max:
        return
    torch.testing.assert_close(graph_output[:, actual_t:], torch.zeros_like(graph_output[:, actual_t:]))
    for name in TOKEN_INPUTS:
        torch.testing.assert_close(graph_grads[name][:, actual_t:], torch.zeros_like(graph_grads[name][:, actual_t:]))


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph capture requires an NVIDIA CUDA device')
@pytest.mark.parametrize('case', [pytest.param(case, id=case.name) for case in GRAPH_CASES])
def test_gdn_varlen_graph_extended_replay(case: GraphCase):
    static_inputs = _make_inputs(case, seed=0, requires_grad=True)
    warm_inputs = _make_inputs(case, seed=0, requires_grad=True)
    cu_seqlens = torch.tensor(case.layouts[0], dtype=torch.long, device=device)
    warm_cu_seqlens = cu_seqlens.clone()
    static_output_grads = _make_output_grads(case, seed=0)
    warm_output_grads = _make_output_grads(case, seed=0)

    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            warm_outputs = _call(warm_inputs, warm_cu_seqlens, case, use_graph=True)
            _backward(warm_outputs, warm_output_grads)
            for name in _grad_names(case):
                warm_value = warm_inputs[name]
                if warm_value is not None:
                    warm_value.grad = None
    torch.cuda.current_stream().wait_stream(warm_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = _call(static_inputs, cu_seqlens, case, use_graph=True)
        _backward(graph_outputs, static_output_grads)

    graph_grads = {}
    for name in _grad_names(case):
        value = static_inputs[name]
        assert value is not None and value.grad is not None
        graph_grads[name] = value.grad

    for seed, offsets in enumerate(case.layouts, start=1):
        replay_inputs = _make_inputs(case, seed=seed, requires_grad=False)
        replay_output_grads = _make_output_grads(case, seed=seed)
        _copy_inputs(static_inputs, replay_inputs)
        cu_seqlens.copy_(torch.tensor(offsets, dtype=torch.long, device=device))
        with torch.no_grad():
            static_output_grads[0].copy_(replay_output_grads[0])
            if static_output_grads[1] is not None:
                static_output_grads[1].copy_(replay_output_grads[1])
            for value in graph_grads.values():
                value.zero_()

        graph.replay()
        torch.cuda.synchronize()

        actual_t = offsets[-1]
        eager_inputs = _slice_inputs(replay_inputs, actual_t, requires_grad=True)
        eager_outputs = _call(eager_inputs, cu_seqlens.clone(), case, use_graph=False)
        _backward(eager_outputs, (replay_output_grads[0][:, :actual_t], replay_output_grads[1]))

        graph_output, graph_final_state = graph_outputs
        eager_output, eager_final_state = eager_outputs
        _assert_finite('graph output', graph_output)
        _assert_finite('graph final state', graph_final_state)
        _assert_finite('eager output', eager_output)
        _assert_finite('eager final state', eager_final_state)
        for name, value in graph_grads.items():
            _assert_finite(f'graph d{name}', value)

        assert_close('o', eager_output, graph_output[:, :actual_t], 0.005)
        if eager_final_state is not None:
            assert graph_final_state is not None
            assert_close('ht', eager_final_state, graph_final_state, 0.005)

        for name in _grad_names(case):
            eager_grad = eager_inputs[name].grad
            assert eager_grad is not None
            graph_grad = graph_grads[name]
            if name in TOKEN_INPUTS:
                graph_grad = graph_grad[:, :actual_t]
            ratio = 0.02 if name in ('g', 'beta', 'A_log', 'dt_bias') else 0.008
            assert_close(f'd{name}', eager_grad, graph_grad, ratio)

        if actual_t < case.t_max:
            assert torch.count_nonzero(replay_inputs['v'][:, actual_t:]) > 0
        _assert_graph_padding(graph_output, graph_grads, actual_t, case)

    del graph
    torch.cuda.empty_cache()


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph capture requires an NVIDIA CUDA device')
@torch.inference_mode()
def test_gdn_external_metadata_replays_balanced_to_ragged():
    case = GraphCase(
        'external-metadata-balanced-to-ragged',
        256,
        4,
        1,
        1,
        32,
        32,
        64,
        torch.float16,
        False,
        False,
        False,
        False,
        False,
        True,
        ((0, 64, 128, 192, 256), (0, 1, 2, 3, 256)),
    )
    nt_max = static_chunk_capacity(case.t_max, case.n_max, case.chunk_size)
    static_inputs = _make_inputs(case, seed=0)
    static_cu = torch.tensor(case.layouts[0], dtype=torch.long, device=device)
    initial_indices, initial_offsets = prepare_chunk_indices_static(static_cu, case.chunk_size, nt_max)
    static_indices = initial_indices.clone()
    static_offsets = initial_offsets.clone()

    warm_inputs = _make_inputs(case, seed=0)
    warm_cu = static_cu.clone()
    warm_indices = static_indices.clone()
    warm_offsets = static_offsets.clone()
    for _ in range(3):
        _call(
            warm_inputs,
            warm_cu,
            case,
            use_graph=False,
            graph_mode='force_graph',
            chunk_indices=warm_indices,
            chunk_offsets=warm_offsets,
            graph_nt_max=nt_max,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_state = _call(
            static_inputs,
            static_cu,
            case,
            use_graph=False,
            graph_mode='force_graph',
            chunk_indices=static_indices,
            chunk_offsets=static_offsets,
            graph_nt_max=nt_max,
        )
    torch.cuda.synchronize()

    output_ptr = graph_output.data_ptr()
    state_ptr = graph_state.data_ptr()
    metadata_ptrs = (static_cu.data_ptr(), static_indices.data_ptr(), static_offsets.data_ptr())
    graph.replay()
    torch.cuda.synchronize()
    eager_a_output, eager_a_state = _call(static_inputs, static_cu.clone(), case, use_graph=False)
    assert_close('capture A output', eager_a_output, graph_output, 0.005)
    assert_close('capture A final state', eager_a_state, graph_state, 0.005)

    replay_inputs = _make_inputs(case, seed=1)
    replay_cu = torch.tensor(case.layouts[1], dtype=torch.long, device=device)
    replay_indices, replay_offsets = prepare_chunk_indices_static(replay_cu, case.chunk_size, nt_max)
    _copy_inputs(static_inputs, replay_inputs)
    static_cu.copy_(replay_cu)
    static_indices.copy_(replay_indices)
    static_offsets.copy_(replay_offsets)
    graph.replay()
    torch.cuda.synchronize()

    assert (graph_output.data_ptr(), graph_state.data_ptr()) == (output_ptr, state_ptr)
    assert (static_cu.data_ptr(), static_indices.data_ptr(), static_offsets.data_ptr()) == metadata_ptrs
    eager_b_output, eager_b_state = _call(replay_inputs, replay_cu, case, use_graph=False)
    assert_close('replay B output', eager_b_output, graph_output, 0.005)
    assert_close('replay B final state', eager_b_state, graph_state, 0.005)

    del graph
    torch.cuda.empty_cache()


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph validation requires an NVIDIA CUDA device')
def test_gdn_graph_rejects_missing_or_malformed_metadata():
    case = GRAPH_CASES[0]
    inputs = _make_inputs(case, seed=0)
    cu_seqlens = torch.tensor(case.layouts[0], dtype=torch.long, device=device)
    common = dict(
        q=inputs['q'],
        k=inputs['k'],
        v=inputs['v'],
        g=inputs['g'],
        beta=inputs['beta'],
        cu_seqlens=cu_seqlens,
        chunk_size=case.chunk_size,
        use_graph=True,
    )

    with pytest.raises(ValueError, match='cu_seqlens'):
        chunk_gated_delta_rule(**{**common, 'cu_seqlens': None})
    with pytest.raises(ValueError, match='chunk_indices'):
        chunk_gated_delta_rule(**{**common, 'chunk_indices': torch.zeros(1, dtype=torch.long, device=device)})
    with pytest.raises(ValueError, match='cu_seqlens_cpu'):
        chunk_gated_delta_rule(**{**common, 'cu_seqlens_cpu': cu_seqlens.clone()})
