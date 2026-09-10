# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import IS_NVIDIA, assert_close, device

T_MAX = 192
N_MAX = 4
H = 2
D = 32
CU_SEQLENS = (
    (0, 48, 96, 144, 192),
    (0, 17, 73, 192, 192),
    (0, 31, 111, 111, 111),
)
TOKEN_INPUTS = ('q', 'k', 'v', 'g', 'beta')
BASE_GRAD_INPUTS = (*TOKEN_INPUTS, 'initial_state')


def _grad_inputs(use_gate_in_kernel: bool) -> tuple[str, ...]:
    return BASE_GRAD_INPUTS + (('A_log', 'dt_bias') if use_gate_in_kernel else ())


def _make_inputs(seed: int, use_gate_in_kernel: bool, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device).manual_seed(seed)
    q = F.normalize(torch.randn(1, T_MAX, H, D, dtype=torch.float32, device=device, generator=generator), dim=-1)
    k = F.normalize(torch.randn(1, T_MAX, H, D, dtype=torch.float32, device=device, generator=generator), dim=-1)
    g = torch.randn(1, T_MAX, H, dtype=torch.float32, device=device, generator=generator)
    if not use_gate_in_kernel:
        g = F.logsigmoid(g)
    inputs = {
        'q': q.to(torch.float16),
        'k': k.to(torch.float16),
        'v': torch.randn(1, T_MAX, H, D, dtype=torch.float16, device=device, generator=generator),
        'g': g,
        'beta': torch.rand(1, T_MAX, H, dtype=torch.float16, device=device, generator=generator),
        'initial_state': torch.randn(N_MAX, H, D, D, dtype=torch.float16, device=device, generator=generator),
        'A_log': torch.randn(H, dtype=torch.float32, device=device, generator=generator),
        'dt_bias': torch.randn(H, dtype=torch.float32, device=device, generator=generator),
    }
    if requires_grad:
        for name in _grad_inputs(use_gate_in_kernel):
            inputs[name].requires_grad_(True)
    return inputs


def _copy_inputs(dst: dict[str, torch.Tensor], src: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name in dst:
            dst[name].copy_(src[name])


def _make_output_grads(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device).manual_seed(seed)
    do = torch.randn(1, T_MAX, H, D, dtype=torch.float16, device=device, generator=generator)
    dht = torch.randn(N_MAX, H, D, D, dtype=torch.float32, device=device, generator=generator)
    return do, dht


def _call(
    inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    use_gate_in_kernel: bool,
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
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=inputs['A_log'] if use_gate_in_kernel else None,
        dt_bias=inputs['dt_bias'] if use_gate_in_kernel else None,
        use_graph=use_graph,
    )


def _eager_reference(
    inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    use_gate_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_t = int(cu_seqlens[-1].item())
    sliced = {
        name: value[:, :actual_t].clone() if value.ndim >= 3 and name != 'initial_state' else value.clone()
        for name, value in inputs.items()
    }
    return _call(sliced, cu_seqlens.clone(), chunk_size, use_gate_in_kernel, use_graph=False)


def _eager_backward_reference(
    inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    chunk_size: int,
    use_gate_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    actual_t = int(cu_seqlens[-1].item())
    eager_inputs = {
        name: (value[:, :actual_t] if name in TOKEN_INPUTS else value).detach().clone()
        for name, value in inputs.items()
    }
    for name in _grad_inputs(use_gate_in_kernel):
        eager_inputs[name].requires_grad_(True)
    eager_o, eager_ht = _call(eager_inputs, cu_seqlens.clone(), chunk_size, use_gate_in_kernel, use_graph=False)
    torch.autograd.backward((eager_o, eager_ht), (do[:, :actual_t], dht))
    grads = {name: eager_inputs[name].grad for name in _grad_inputs(use_gate_in_kernel)}
    return eager_o, eager_ht, grads


@pytest.mark.skipif(not IS_NVIDIA, reason="CUDA Graph capture requires an NVIDIA CUDA device")
@pytest.mark.parametrize(
    ('chunk_size', 'use_gate_in_kernel'),
    [(16, False), (32, False), (64, False), (64, True)],
)
@torch.inference_mode()
def test_gdn_varlen_graph_forward_replay(chunk_size: int, use_gate_in_kernel: bool):
    static_inputs = _make_inputs(seed=0, use_gate_in_kernel=use_gate_in_kernel)
    cu_seqlens = torch.tensor(CU_SEQLENS[0], dtype=torch.long, device=device)

    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            _call(static_inputs, cu_seqlens, chunk_size, use_gate_in_kernel, use_graph=True)
    torch.cuda.current_stream().wait_stream(warm_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_o, graph_ht = _call(static_inputs, cu_seqlens, chunk_size, use_gate_in_kernel, use_graph=True)

    for seed, offsets in enumerate(CU_SEQLENS, start=1):
        replay_inputs = _make_inputs(seed=seed, use_gate_in_kernel=use_gate_in_kernel)
        _copy_inputs(static_inputs, replay_inputs)
        cu_seqlens.copy_(torch.tensor(offsets, dtype=torch.long, device=device))
        graph.replay()
        torch.cuda.synchronize()

        actual_t = offsets[-1]
        eager_o, eager_ht = _eager_reference(replay_inputs, cu_seqlens, chunk_size, use_gate_in_kernel)
        torch.testing.assert_close(graph_o[:, :actual_t], eager_o, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(graph_ht, eager_ht, rtol=5e-3, atol=5e-3)
        if actual_t < T_MAX:
            assert torch.count_nonzero(replay_inputs['v'][:, actual_t:]) > 0
            torch.testing.assert_close(graph_o[:, actual_t:], torch.zeros_like(graph_o[:, actual_t:]))


@pytest.mark.skipif(not IS_NVIDIA, reason="CUDA Graph capture requires an NVIDIA CUDA device")
@pytest.mark.parametrize(
    ('chunk_size', 'use_gate_in_kernel'),
    [(16, False), (32, False), (64, False), (64, True)],
)
def test_gdn_varlen_graph_backward_replay(chunk_size: int, use_gate_in_kernel: bool):
    static_inputs = _make_inputs(seed=0, use_gate_in_kernel=use_gate_in_kernel, requires_grad=True)
    warm_inputs = _make_inputs(seed=0, use_gate_in_kernel=use_gate_in_kernel, requires_grad=True)
    cu_seqlens = torch.tensor(CU_SEQLENS[0], dtype=torch.long, device=device)
    static_do, static_dht = _make_output_grads(seed=0)
    grad_names = _grad_inputs(use_gate_in_kernel)

    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            warm_o, warm_ht = _call(warm_inputs, cu_seqlens, chunk_size, use_gate_in_kernel, use_graph=True)
            torch.autograd.backward((warm_o, warm_ht), (static_do, static_dht))
            for name in grad_names:
                warm_inputs[name].grad = None
    torch.cuda.current_stream().wait_stream(warm_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_o, graph_ht = _call(static_inputs, cu_seqlens, chunk_size, use_gate_in_kernel, use_graph=True)
        torch.autograd.backward((graph_o, graph_ht), (static_do, static_dht))
    graph_grads = {name: static_inputs[name].grad for name in grad_names}

    grad_tolerances = {
        'q': 0.008,
        'k': 0.008,
        'v': 0.008,
        'g': 0.02,
        'beta': 0.02,
        'initial_state': 0.008,
        'A_log': 0.02,
        'dt_bias': 0.02,
    }
    for seed, offsets in enumerate(CU_SEQLENS, start=1):
        replay_inputs = _make_inputs(seed=seed, use_gate_in_kernel=use_gate_in_kernel)
        replay_do, replay_dht = _make_output_grads(seed=seed)
        _copy_inputs(static_inputs, replay_inputs)
        cu_seqlens.copy_(torch.tensor(offsets, dtype=torch.long, device=device))
        static_do.copy_(replay_do)
        static_dht.copy_(replay_dht)
        for grad in graph_grads.values():
            grad.zero_()
        graph.replay()
        torch.cuda.synchronize()

        actual_t = offsets[-1]
        eager_o, eager_ht, eager_grads = _eager_backward_reference(
            replay_inputs,
            cu_seqlens,
            replay_do,
            replay_dht,
            chunk_size,
            use_gate_in_kernel,
        )
        assert_close('o', eager_o, graph_o[:, :actual_t], 0.005)
        assert_close('ht', eager_ht, graph_ht, 0.005)
        for name in grad_names:
            graph_grad = graph_grads[name][:, :actual_t] if name in TOKEN_INPUTS else graph_grads[name]
            assert_close(f'd{name}', eager_grads[name], graph_grad, grad_tolerances[name])

        if actual_t < T_MAX:
            assert torch.count_nonzero(replay_inputs['v'][:, actual_t:]) > 0
            torch.testing.assert_close(graph_o[:, actual_t:], torch.zeros_like(graph_o[:, actual_t:]))
            for name in TOKEN_INPUTS:
                tail = graph_grads[name][:, actual_t:]
                torch.testing.assert_close(tail, torch.zeros_like(tail))
