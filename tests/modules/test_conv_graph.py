# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import gc

import pytest
import torch

from fla.modules.convolution import causal_conv1d
from fla.ops.utils.graph import static_chunk_capacity
from fla.ops.utils.index import prepare_chunk_indices_static
from fla.utils import IS_NVIDIA, assert_close, device

T_MAX = 128
N_MAX = 4
D = 32
WIDTH = 4
CHUNK_SIZE = 16
NT_MAX = static_chunk_capacity(T_MAX, N_MAX, CHUNK_SIZE)
LAYOUTS = (
    (0, 32, 64, 96, 128),
    (0, 1, 17, 80, 80),
    (0, 16, 64, 64, 128),
    (0, 0, 0, 0, 128),
)


def _make_tensor(shape: tuple[int, ...], dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device=device, generator=generator)


def _zero_grad(values: tuple[torch.Tensor, ...]) -> None:
    for value in values:
        if value.grad is not None:
            value.grad.zero_()


def _update_static_metadata(
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    offsets: tuple[int, ...],
) -> None:
    cu_seqlens.copy_(torch.tensor(offsets, dtype=cu_seqlens.dtype, device=device))
    fresh_indices, _ = prepare_chunk_indices_static(cu_seqlens, CHUNK_SIZE, NT_MAX)
    chunk_indices.copy_(fresh_indices)


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph capture requires an NVIDIA CUDA device')
def test_causal_conv_graph_replay_forward_backward():
    torch.manual_seed(42)
    dtype = torch.float16
    generator = torch.Generator(device=device).manual_seed(3)
    static_x = _make_tensor((1, T_MAX, D), dtype, generator).requires_grad_(True)
    static_weight = _make_tensor((D, WIDTH), dtype, generator).requires_grad_(True)
    static_bias = _make_tensor((D,), dtype, generator).requires_grad_(True)
    static_residual = _make_tensor((1, T_MAX, D), dtype, generator).requires_grad_(True)
    static_initial = _make_tensor((N_MAX, D, WIDTH), dtype, generator).requires_grad_(True)
    static_do = _make_tensor((1, T_MAX, D), dtype, generator)
    static_dht = _make_tensor((N_MAX, D, WIDTH), dtype, generator)
    static_cu = torch.tensor(LAYOUTS[0], dtype=torch.long, device=device)
    initial_indices, _ = prepare_chunk_indices_static(static_cu, CHUNK_SIZE, NT_MAX)
    static_indices = initial_indices.clone()
    values = (static_x, static_weight, static_bias, static_residual, static_initial)

    # Keep warmup autograd nodes off the graph capture by releasing each temporary graph.
    for _ in range(2):
        warm_output, warm_state = causal_conv1d(
            static_x,
            static_weight,
            static_bias,
            static_residual,
            static_initial,
            True,
            'silu',
            cu_seqlens=static_cu,
            chunk_indices=static_indices,
            chunk_size=CHUNK_SIZE,
            use_graph=True,
            graph_nt_max=NT_MAX,
        )
        torch.autograd.backward((warm_output, warm_state), (static_do, static_dht))
        del warm_output, warm_state
        for value in values:
            value.grad = None
        gc.collect()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_state = causal_conv1d(
            static_x,
            static_weight,
            static_bias,
            static_residual,
            static_initial,
            True,
            'silu',
            cu_seqlens=static_cu,
            chunk_indices=static_indices,
            chunk_size=CHUNK_SIZE,
            use_graph=True,
            graph_nt_max=NT_MAX,
        )
        torch.autograd.backward((graph_output, graph_state), (static_do, static_dht))
    torch.cuda.synchronize()

    output_ptr = graph_output.data_ptr()
    metadata_ptrs = (static_cu.data_ptr(), static_indices.data_ptr())
    graph_grads = {name: value.grad for name, value in zip(('x', 'weight', 'bias', 'residual', 'initial'), values)}
    assert all(value is not None for value in graph_grads.values())

    for seed, offsets in enumerate(LAYOUTS, start=1):
        replay_generator = torch.Generator(device=device).manual_seed(seed)
        replay_values = (
            _make_tensor(static_x.shape, dtype, replay_generator),
            _make_tensor(static_weight.shape, dtype, replay_generator),
            _make_tensor(static_bias.shape, dtype, replay_generator),
            _make_tensor(static_residual.shape, dtype, replay_generator),
            _make_tensor(static_initial.shape, dtype, replay_generator),
        )
        replay_do = _make_tensor(static_do.shape, dtype, replay_generator)
        replay_dht = _make_tensor(static_dht.shape, dtype, replay_generator)
        _update_static_metadata(static_cu, static_indices, offsets)
        with torch.no_grad():
            for destination, source in zip(values, replay_values):
                destination.copy_(source)
            static_do.copy_(replay_do)
            static_dht.copy_(replay_dht)
        _zero_grad(values)

        graph.replay()
        torch.cuda.synchronize()
        assert (graph_output.data_ptr(), static_cu.data_ptr(), static_indices.data_ptr()) == (
            output_ptr,
            *metadata_ptrs,
        )

        actual_t = offsets[-1]
        eager_x = replay_values[0][:, :actual_t].detach().clone().requires_grad_(True)
        eager_weight = replay_values[1].detach().clone().requires_grad_(True)
        eager_bias = replay_values[2].detach().clone().requires_grad_(True)
        eager_residual = replay_values[3][:, :actual_t].detach().clone().requires_grad_(True)
        eager_initial = replay_values[4].detach().clone().requires_grad_(True)
        eager_cu = torch.tensor(offsets, dtype=torch.long, device=device)
        eager_output, eager_state = causal_conv1d(
            eager_x,
            eager_weight,
            eager_bias,
            eager_residual,
            eager_initial,
            True,
            'silu',
            cu_seqlens=eager_cu,
            chunk_size=CHUNK_SIZE,
            use_graph=False,
        )
        torch.autograd.backward(
            (eager_output, eager_state),
            (replay_do[:, :actual_t], replay_dht),
        )
        torch.cuda.synchronize()

        assert_close('conv output', eager_output, graph_output[:, :actual_t], 0.02)
        assert_close('conv final state', eager_state, graph_state, 0.02)
        eager_values = (eager_x, eager_weight, eager_bias, eager_residual, eager_initial)
        for name, eager_value, graph_value in zip(('x', 'weight', 'bias', 'residual', 'initial'), eager_values, values):
            compared_graph_grad = graph_grads[name]
            if name in ('x', 'residual'):
                compared_graph_grad = compared_graph_grad[:, :actual_t]
            assert_close(f'conv gradient {name}', eager_value.grad, compared_graph_grad, 0.03)
        if actual_t < T_MAX:
            torch.testing.assert_close(graph_output[:, actual_t:], torch.zeros_like(graph_output[:, actual_t:]))
            torch.testing.assert_close(graph_grads['x'][:, actual_t:], torch.zeros_like(graph_grads['x'][:, actual_t:]))
            torch.testing.assert_close(
                graph_grads['residual'][:, actual_t:],
                torch.zeros_like(graph_grads['residual'][:, actual_t:]),
            )
        assert torch.isfinite(graph_output).all()
        assert torch.isfinite(graph_state).all()

    del graph
    torch.cuda.empty_cache()
