# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import inspect
from types import SimpleNamespace

import pytest
import torch

from fla.ops.cp import chunk_delta_h, context
from fla.ops.cp.context import build_cp_context, get_cp_cu_seqlens
from fla.utils import IS_NPU, device


@pytest.mark.parametrize(
    ('total_tokens', 'world_size'),
    [
        pytest.param(10, 3, id='remainder'),
        pytest.param(14, 4, id='remainder-varlen'),
    ],
)
def test_get_cp_cu_seqlens_invalid_total(total_tokens: int, world_size: int):
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match=rf"`total_tokens` \({total_tokens}\) must be divisible by `world_size` \({world_size}\)",
    ):
        get_cp_cu_seqlens(cu_seqlens, world_size=world_size, rank=0)


def test_get_cp_cu_seqlens_empty_partition():
    total_tokens = 2
    world_size = 3
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match=rf"`total_tokens` \({total_tokens}\) must be at least `world_size` \({world_size}\)",
    ):
        get_cp_cu_seqlens(cu_seqlens, world_size=world_size, rank=0)


def test_get_cp_cu_seqlens_varlen():
    context = get_cp_cu_seqlens(
        cu_seqlens=torch.tensor([0, 5, 12], dtype=torch.long),
        world_size=3,
        rank=1,
    )

    assert context.cu_seqlens_cpu.tolist() == [0, 1, 4]
    assert context.pre_num_conv_tokens == 4
    assert context.pre_num_ranks == 1
    assert not context.is_first_rank
    assert context.post_num_ranks == 1
    assert not context.is_last_rank


@pytest.mark.parametrize('use_tf32x3', [False, True])
def test_precision_keyword(use_tf32x3: bool, monkeypatch):
    monkeypatch.setattr(context.dist, 'get_world_size', lambda group: 2)
    monkeypatch.setattr(context.dist, 'get_rank', lambda group: 0)
    kwargs = dict(cu_seqlens=torch.tensor([0, 256]), group=object(), use_tf32x3_affine_chain=use_tf32x3)
    if use_tf32x3:
        cp_context = build_cp_context(**kwargs)
        assert not hasattr(cp_context, 'use_tf32x3_affine_chain')
    else:
        with pytest.raises(ValueError, match='use_tf32x3_affine_chain=False'):
            build_cp_context(**kwargs)


@pytest.mark.parametrize(
    ('is_nvidia', 'is_npu', 'device_name', 'capability', 'supported'),
    [
        pytest.param(True, False, 'cuda:1', (7, 5), False, id='turing'),
        pytest.param(True, False, 'cuda:1', (8, 0), True, id='ampere'),
        pytest.param(False, False, 'cuda:1', (9, 0), False, id='amd'),
        pytest.param(False, False, 'cpu', (0, 0), False, id='cpu'),
        pytest.param(False, True, 'npu:1', None, True, id='ascend'),
    ],
)
def test_device_requirement(is_nvidia, is_npu, device_name, capability, supported, monkeypatch):
    monkeypatch.setattr(context, 'IS_NVIDIA', is_nvidia)
    monkeypatch.setattr(context, 'IS_NPU', is_npu)
    device = SimpleNamespace(type='npu', index=1) if is_npu else torch.device(device_name)
    checked_devices = []

    def get_capability(device_index):
        checked_devices.append(device_index)
        return capability

    monkeypatch.setattr(context, 'get_device_capability', get_capability)
    if supported:
        context._check_cp_device(device)
    else:
        with pytest.raises(RuntimeError, match='compute capability >= 8.0'):
            context._check_cp_device(device)
    assert checked_devices == ([1] if is_nvidia else [])


@pytest.mark.parametrize('operation', ['forward', 'backward', 'compress', 'expand'])
def test_zigzag_graph_requires_npu(operation, monkeypatch):
    monkeypatch.setattr(chunk_delta_h, 'IS_NPU', False)
    monkeypatch.setattr(chunk_delta_h, '_check_cp_device', lambda device: None)
    monkeypatch.setattr(chunk_delta_h.dist, 'get_rank', lambda group: 0)
    cu = torch.tensor([0, 1, 2, 3, 4])
    cp_context = SimpleNamespace(
        group=object(),
        layout='zigzag',
        front_num_seqs=2,
        cu_seqlens=cu,
        pre_num_ranks_dev=torch.zeros(2, dtype=torch.int32),
        post_num_ranks_dev=torch.zeros(2, dtype=torch.int32),
    )
    x = torch.empty(1, 4, 1, 16)
    with pytest.raises(NotImplementedError, match='zigzag CP graph.*Ascend NPU'):
        if operation == 'forward':
            inspect.unwrap(chunk_delta_h.chunk_gated_delta_rule_fwd_h_pre_process)(
                k=x, w=x, u=x, cu_seqlens=cu, context=cp_context, use_graph=True,
            )
        elif operation == 'backward':
            inspect.unwrap(chunk_delta_h.chunk_gated_delta_rule_bwd_dhu_pre_process)(
                q=x, k=x, w=x, do=x, dv=x, cu_seqlens=cu, context=cp_context, use_graph=True,
            )
        elif operation == 'compress':
            chunk_delta_h.compress_h0(torch.empty(4, 1, 16, 16), context=cp_context, use_graph=True)
        else:
            chunk_delta_h.expand_h0(torch.empty(2, 1, 16, 16), context=cp_context, use_graph=True)


@pytest.mark.parametrize(('layout', 'use_graph'), [('contiguous', False), ('contiguous', True), ('zigzag', False)])
def test_state_compression_non_npu(layout, use_graph, monkeypatch):
    monkeypatch.setattr(chunk_delta_h, 'IS_NPU', False)
    cp_context = SimpleNamespace(layout=layout, front_num_seqs=2, cu_seqlens=torch.tensor([0, 1, 2, 3, 4]))
    state = torch.arange(4).view(4, 1, 1, 1)
    indices = [0] if layout == 'contiguous' else [0, 2]
    compressed = chunk_delta_h.compress_h0(state, context=cp_context, use_graph=use_graph)
    torch.testing.assert_close(compressed, state[indices])
    expanded = chunk_delta_h.expand_h0(compressed, context=cp_context, use_graph=use_graph)
    expected = torch.zeros_like(state)
    expected[indices] = state[indices]
    torch.testing.assert_close(expanded, expected)


@pytest.mark.skipif(not IS_NPU, reason='requires Ascend graph metadata kernels')
@pytest.mark.parametrize(
    ('cu_dtype', 'capacity', 'total', 'stride'),
    [
        pytest.param(torch.int32, 2, 128, 1, id='int32-two-sequences'),
        pytest.param(torch.int64, 8, 128, 1, id='int64-tail-capacity'),
        pytest.param(torch.int64, 1024, 1 << 34, 1, id='int64-reduction-tile-large-offsets'),
        pytest.param(torch.int32, 1025, 128, 2, id='int32-strided-two-tiles'),
        pytest.param(torch.int64, 32768, 128, 1, id='int64-large-capacity'),
    ],
)
def test_zigzag_graph_metadata(cu_dtype, capacity, total, stride):
    """Replay sequence boundary changes across reduction tiles and integer layouts."""
    storage = torch.empty((capacity + 1) * stride, device=device, dtype=cu_dtype)
    cu = storage[::stride]

    def refresh(offsets):
        cpu = torch.full((capacity + 1,), offsets[-1], dtype=cu_dtype)
        cpu[:len(offsets)] = torch.tensor(offsets, dtype=cu_dtype)
        cu.copy_(cpu)
        return cpu

    refresh([0, total // 2, total])
    for _ in range(3):
        chunk_delta_h._zigzag_graph_metadata(cu)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = chunk_delta_h._zigzag_graph_metadata(cu)
    tensors = (actual[0], actual[1], *actual[2], *actual[3])
    assert len({x.untyped_storage().data_ptr() for x in tensors}) == 1
    assert actual[2][0].data_ptr() - actual[0].data_ptr() == 4 * 8
    pointers = tuple(x.data_ptr() for x in tensors)
    layouts = [[0, total // 2, total], [0, total // 4, total // 2]]
    if capacity >= 4:
        layouts += [[0, 1, total // 2, total // 2 + 1, total], [0, total // 4, total // 2, total]]
    if capacity >= 1023:
        active = capacity - capacity % 2
        layouts.append(list(range(active + 1)))
    for offsets in layouts + [[0], layouts[0]]:
        cpu = refresh(offsets)
        graph.replay()
        torch.npu.synchronize()
        front = int((cpu[:-1] < cpu[-1] // 2).sum())
        count = int((cpu[1:] > cpu[:-1]).sum())
        first, last = torch.tensor([0, front]), torch.tensor([front - 1, count - 1])
        windows = tuple(cpu[index:index + 2] if 0 <= index < capacity else torch.zeros(2, dtype=cu_dtype)
                        for index in (*first, *last))
        expected = (first, last, *windows)
        for got, want in zip(tensors, expected):
            torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
        assert tuple(x.data_ptr() for x in tensors) == pointers
