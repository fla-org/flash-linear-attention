# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import gc

import pytest
import torch

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.ops.utils.graph import static_chunk_capacity
from fla.ops.utils.index import prepare_chunk_indices_static
from fla.utils import IS_NVIDIA, assert_close, device

T_MAX = 128
N_MAX = 4
HIDDEN_SIZE = 128
HEAD_DIM = 16
NUM_HEADS = 4
EXPAND_V = 2
CHUNK_SIZE = 16
NT_MAX = static_chunk_capacity(T_MAX, N_MAX, CHUNK_SIZE)
LAYOUTS = (
    (0, 32, 64, 96, 128),
    (0, 1, 17, 80, 80),
    (0, 16, 64, 64, 128),
    (0, 0, 0, 0, 128),
)


def _make_layer(dtype: torch.dtype) -> GatedDeltaNet:
    return GatedDeltaNet(
        hidden_size=HIDDEN_SIZE,
        head_dim=HEAD_DIM,
        num_heads=NUM_HEADS,
        expand_v=EXPAND_V,
        mode='chunk',
        use_short_conv=True,
        conv_size=4,
        conv_bias=True,
    ).to(device=device, dtype=dtype).train()


def _make_tensor(shape: tuple[int, ...], dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device=device, generator=generator)


def _zero_grad_buffers(layer: GatedDeltaNet, x: torch.Tensor) -> None:
    if x.grad is not None:
        x.grad.zero_()
    for parameter in layer.parameters():
        if parameter.grad is not None:
            parameter.grad.zero_()


def _update_static_metadata(
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    offsets: tuple[int, ...],
) -> None:
    cu_seqlens.copy_(torch.tensor(offsets, dtype=cu_seqlens.dtype, device=device))
    fresh_indices, fresh_offsets = prepare_chunk_indices_static(cu_seqlens, CHUNK_SIZE, NT_MAX)
    chunk_indices.copy_(fresh_indices)
    chunk_offsets.copy_(fresh_offsets)


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph capture requires an NVIDIA CUDA device')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
def test_gated_deltanet_layer_graph_replay_forward_backward(dtype: torch.dtype):
    torch.manual_seed(42)
    graph_layer = _make_layer(dtype)
    eager_layer = _make_layer(dtype)
    eager_layer.load_state_dict(graph_layer.state_dict())

    generator = torch.Generator(device=device).manual_seed(7)
    static_x = _make_tensor((1, T_MAX, HIDDEN_SIZE), dtype, generator).requires_grad_(True)
    static_do = _make_tensor((1, T_MAX, HIDDEN_SIZE), dtype, generator)
    static_cu = torch.tensor(LAYOUTS[0], dtype=torch.long, device=device)
    initial_indices, initial_offsets = prepare_chunk_indices_static(static_cu, CHUNK_SIZE, NT_MAX)
    static_indices = initial_indices.clone()
    static_offsets = initial_offsets.clone()

    # Warm up on the capture stream and release the temporary autograd graphs.
    for _ in range(2):
        warm_output = graph_layer(
            static_x,
            cu_seqlens=static_cu,
            chunk_indices=static_indices,
            chunk_offsets=static_offsets,
            use_graph=True,
            graph_t_max=T_MAX,
            graph_n_max=N_MAX,
            graph_nt_max=NT_MAX,
            chunk_size=CHUNK_SIZE,
        )[0]
        torch.autograd.backward(warm_output, static_do)
        del warm_output
        for parameter in graph_layer.parameters():
            parameter.grad = None
        static_x.grad = None
        gc.collect()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = graph_layer(
            static_x,
            cu_seqlens=static_cu,
            chunk_indices=static_indices,
            chunk_offsets=static_offsets,
            use_graph=True,
            graph_t_max=T_MAX,
            graph_n_max=N_MAX,
            graph_nt_max=NT_MAX,
            chunk_size=CHUNK_SIZE,
        )[0]
        torch.autograd.backward(graph_output, static_do)
    torch.cuda.synchronize()

    graph_output_ptr = graph_output.data_ptr()
    graph_metadata_ptrs = (static_indices.data_ptr(), static_offsets.data_ptr(), static_cu.data_ptr())
    graph_grads = {name: parameter.grad for name, parameter in graph_layer.named_parameters()}
    assert static_x.grad is not None
    assert all(gradient is not None for gradient in graph_grads.values())

    for seed, offsets in enumerate(LAYOUTS, start=1):
        replay_generator = torch.Generator(device=device).manual_seed(seed)
        replay_x = _make_tensor((1, T_MAX, HIDDEN_SIZE), dtype, replay_generator)
        replay_do = _make_tensor((1, T_MAX, HIDDEN_SIZE), dtype, replay_generator)
        _update_static_metadata(static_cu, static_indices, static_offsets, offsets)
        with torch.no_grad():
            static_x.copy_(replay_x)
            static_do.copy_(replay_do)
        _zero_grad_buffers(graph_layer, static_x)

        graph.replay()
        torch.cuda.synchronize()
        assert (graph_output.data_ptr(), static_indices.data_ptr(), static_offsets.data_ptr(), static_cu.data_ptr()) == (
            graph_output_ptr,
            *graph_metadata_ptrs,
        )

        actual_t = offsets[-1]
        eager_x = replay_x[:, :actual_t].detach().clone().requires_grad_(True)
        eager_do = replay_do[:, :actual_t]
        eager_cu = torch.tensor(offsets, dtype=torch.long, device=device)
        eager_layer.zero_grad(set_to_none=True)
        eager_output = eager_layer(eager_x, cu_seqlens=eager_cu, use_graph=False)[0]
        torch.autograd.backward(eager_output, eager_do)
        torch.cuda.synchronize()

        assert_close('layer output', eager_output, graph_output[:, :actual_t], 0.02)
        assert_close('layer input gradient', eager_x.grad, static_x.grad[:, :actual_t], 0.03)
        for name, parameter in eager_layer.named_parameters():
            assert_close(f'layer gradient {name}', parameter.grad, graph_grads[name], 0.04)
        if actual_t < T_MAX:
            torch.testing.assert_close(graph_output[:, actual_t:], torch.zeros_like(graph_output[:, actual_t:]))
            torch.testing.assert_close(static_x.grad[:, actual_t:], torch.zeros_like(static_x.grad[:, actual_t:]))
        assert torch.isfinite(graph_output).all()
        assert torch.isfinite(static_x.grad).all()

    del graph
    torch.cuda.empty_cache()


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph routing requires an NVIDIA CUDA device')
@torch.inference_mode()
def test_gated_deltanet_auto_fallback_rebuilds_metadata():
    layer = GatedDeltaNet(
        hidden_size=32,
        head_dim=16,
        num_heads=2,
        expand_v=1,
        mode='chunk',
        use_short_conv=False,
    ).to(device=device, dtype=torch.float16).eval()
    generator = torch.Generator(device=device).manual_seed(13)
    static_x = torch.randn(1, T_MAX, 32, dtype=torch.float16, device=device, generator=generator)
    static_cu = torch.tensor((0, 1, 1, 1, 1), dtype=torch.long, device=device)
    static_indices, static_offsets = prepare_chunk_indices_static(static_cu, CHUNK_SIZE, NT_MAX)

    routed = layer(
        static_x,
        cu_seqlens=static_cu,
        cu_seqlens_cpu=static_cu.cpu(),
        graph_mode='auto',
        graph_t_max=T_MAX,
        graph_n_max=N_MAX,
        graph_nt_max=NT_MAX,
        chunk_indices=static_indices,
        chunk_offsets=static_offsets,
        chunk_size=CHUNK_SIZE,
    )[0]
    reference = layer(
        static_x[:, :1],
        cu_seqlens=torch.tensor((0, 1, 1, 1, 1), dtype=torch.long, device=device),
        graph_mode='eager',
        chunk_size=CHUNK_SIZE,
    )[0]
    torch.testing.assert_close(routed[:, :1], reference, atol=0.02, rtol=0.02)
    # eager fallback defines only the packed tokens addressed by cu_seqlens
    assert torch.isfinite(routed[:, :1]).all()


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph routing requires an NVIDIA CUDA device')
@torch.inference_mode()
def test_gated_deltanet_force_graph_requires_external_metadata():
    layer = _make_layer(torch.float16).eval()
    hidden_states = torch.randn(1, T_MAX, HIDDEN_SIZE, dtype=torch.float16, device=device)
    cu_seqlens = torch.tensor(LAYOUTS[0], dtype=torch.long, device=device)
    with pytest.raises(ValueError, match='caller-provided fixed'):
        layer(
            hidden_states,
            cu_seqlens=cu_seqlens,
            graph_mode='force_graph',
            graph_t_max=T_MAX,
            graph_n_max=N_MAX,
            graph_nt_max=NT_MAX,
            chunk_size=CHUNK_SIZE,
        )
