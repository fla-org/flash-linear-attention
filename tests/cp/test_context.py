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


@pytest.mark.parametrize('operation', ['forward', 'backward', 'prepare'])
def test_zigzag_graph_requires_npu(operation, monkeypatch):
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
    with pytest.raises(NotImplementedError, match='zigzag CP'):
        if operation == 'forward':
            inspect.unwrap(chunk_delta_h.chunk_gated_delta_rule_fwd_h_pre_process)(
                k=x, w=x, u=x, cu_seqlens=cu, context=cp_context, use_graph=True,
            )
        elif operation == 'backward':
            inspect.unwrap(chunk_delta_h.chunk_gated_delta_rule_bwd_dhu_pre_process)(
                q=x, k=x, w=x, do=x, dv=x, cu_seqlens=cu, context=cp_context, use_graph=True,
            )
        else:
            chunk_delta_h.prepare_cp_graph_context(cu, cp_context)


@pytest.mark.parametrize(
    ('layout', 'use_graph', 'offsets', 'indices'),
    [('contiguous', False, [0, 1, 2, 3, 4], [0]), ('contiguous', True, [0, 1, 2, 3, 4], [0]),
     ('zigzag', False, [0, 1, 2, 3, 4], [0, 2]), ('contiguous', True, [0, 0, 2, 2, 4], [1])],
)
def test_state_compression_non_npu(layout, use_graph, offsets, indices):
    cp_context = SimpleNamespace(layout=layout, front_num_seqs=2, cu_seqlens=torch.tensor(offsets))
    if use_graph:
        cp_context = chunk_delta_h.prepare_cp_graph_context(cp_context.cu_seqlens, cp_context)
    state = torch.arange(4).view(4, 1, 1, 1)
    compressed = chunk_delta_h.compress_h0(state, context=cp_context)
    torch.testing.assert_close(compressed, state[indices])
    expanded = chunk_delta_h.expand_h0(compressed, context=cp_context)
    expected = torch.zeros_like(state)
    expected[indices] = state[indices]
    torch.testing.assert_close(expanded, expected)


@pytest.mark.parametrize('cu_dtype', [torch.int32, torch.int64])
def test_contiguous_graph_boundaries_non_npu(cu_dtype, monkeypatch):
    """Check shared-wrapper routing without requiring a CUDA communicator or kernels."""
    monkeypatch.setattr(chunk_delta_h, '_check_cp_device', lambda device: None)
    monkeypatch.setattr(chunk_delta_h.dist, 'get_rank', lambda group: 1)
    monkeypatch.setattr(chunk_delta_h, 'all_gather_into_tensor', lambda x, group: (x, None))
    calls = {}
    monkeypatch.setattr(chunk_delta_h, 'chunk_cp_fwd_h', lambda **kwargs: calls.update(fwd=kwargs))
    monkeypatch.setattr(chunk_delta_h, 'chunk_cp_bwd_dh', lambda **kwargs: calls.update(bwd=kwargs))
    monkeypatch.setattr(chunk_delta_h, 'chunk_cp_merge', lambda **kwargs: calls.update(merge=kwargs))
    storage = torch.tensor([0, -1, 0, -1, 2, -1, 2, -1, 4, -1, 4, -1], dtype=cu_dtype)
    # leading, internal and trailing empty rows, plus a non-unit stride
    cu = storage[::2]
    cp_context = SimpleNamespace(
        group=object(), layout='contiguous', cu_seqlens=cu,
        pre_num_ranks_dev=torch.ones(1, dtype=torch.int32),
        post_num_ranks_dev=torch.ones(1, dtype=torch.int32),
    )
    x = torch.empty(1, 4, 1, 16)
    for first_row, last_row, first_window, last_window in (
        (1, 3, [0, 2], [2, 4]),
        (0, -1, [0, 0], [0, 0]),
    ):
        if last_row < 0:
            cu.zero_()
            cp_context.pre_num_ranks_dev.zero_()
            cp_context.post_num_ranks_dev.zero_()
        initial = inspect.unwrap(chunk_delta_h.chunk_gated_delta_rule_fwd_h_pre_process)(
            k=x, w=x, u=x, cu_seqlens=cu, context=cp_context, use_graph=True,
        )
        torch.testing.assert_close(calls['fwd']['cu_seqlens'], torch.tensor(last_window, dtype=cu_dtype))
        torch.testing.assert_close(calls['merge']['h_seq_idx'], torch.tensor([first_row]))
        assert calls['merge']['h'] is initial
        final, _ = inspect.unwrap(chunk_delta_h.chunk_gated_delta_rule_bwd_dhu_pre_process)(
            q=x, k=x, w=x, do=x, dv=x, cu_seqlens=cu, context=cp_context, use_graph=True,
        )
        torch.testing.assert_close(calls['bwd']['cu_seqlens'], torch.tensor(first_window, dtype=cu_dtype))
        torch.testing.assert_close(calls['merge']['h_seq_idx'], torch.tensor([last_row]))
        assert calls['merge']['h'] is final


@pytest.mark.skipif(not IS_NPU, reason='requires Ascend graph metadata kernels')
@pytest.mark.parametrize(
    ('layout', 'cu_dtype', 'capacity', 'total', 'stride'),
    [
        pytest.param('contiguous', torch.int32, 8, 128, 1, id='contiguous-int32-tail-capacity'),
        pytest.param('zigzag', torch.int64, 8, 128, 1, id='zigzag-int64-tail-capacity'),
        pytest.param('zigzag', torch.int64, 1024, 1 << 34, 1, id='zigzag-int64-reduction-tile-large-offsets'),
        pytest.param('contiguous', torch.int32, 1025, 128, 2, id='contiguous-int32-strided-two-tiles'),
        pytest.param('zigzag', torch.int64, 32768, 128, 1, id='zigzag-int64-large-capacity'),
    ],
)
def test_cp_graph_metadata(layout, cu_dtype, capacity, total, stride):
    """Replay sequence boundary changes across reduction tiles and integer layouts."""
    from fla.ops.cp.backends.triton_ascend.chunk_delta_h import prepare_cp_graph_metadata

    storage = torch.empty((capacity + 1) * stride, device=device, dtype=cu_dtype)
    cu = storage[::stride]

    def refresh(offsets):
        cpu = torch.full((capacity + 1,), offsets[-1], dtype=cu_dtype)
        cpu[:len(offsets)] = torch.tensor(offsets, dtype=cu_dtype)
        cu.copy_(cpu)
        return cpu

    refresh([0, total // 2, total])
    for _ in range(3):
        prepare_cp_graph_metadata(cu, layout=layout)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = prepare_cp_graph_metadata(cu, layout=layout)
    tensors = (actual[0], actual[1], *actual[2], *actual[3])
    assert len({x.untyped_storage().data_ptr() for x in tensors}) == 1
    num_parts = 2 if layout == 'zigzag' else 1
    assert actual[2][0].data_ptr() - actual[0].data_ptr() == 2 * num_parts * 8
    pointers = tuple(x.data_ptr() for x in tensors)
    layouts = [[0, total // 2, total], [0, total // 4, total // 2]]
    if capacity >= 4:
        layouts += [
            [0, 1, total // 2, total // 2 + 1, total], [0, total // 4, total // 2, total],
            [0, 0, total // 2, total], [0, total // 2, total // 2, total],
        ]
    if capacity >= 6:
        layouts.append([0, 0, total // 4, total // 4, total // 2, total // 2, total])
    if capacity >= 1023:
        active = capacity - capacity % 2
        layouts.append(list(range(active + 1)))
    for offsets in layouts + [[0], layouts[0]]:
        cpu = refresh(offsets)
        graph.replay()
        torch.npu.synchronize()
        boundaries = cpu.tolist()
        intervals = [(index, start) for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])) if end > start]
        parts = ([[index for index, start in intervals if (start < boundaries[-1] // 2) == front]
                  for front in (True, False)] if layout == 'zigzag' else [[index for index, _ in intervals]])
        first = torch.tensor([part[0] if part else 0 for part in parts])
        last = torch.tensor([part[-1] if part else -1 for part in parts])
        windows = tuple(cpu[index:index + 2] if 0 <= index < capacity else torch.zeros(2, dtype=cu_dtype)
                        for index in (*first, *last))
        expected = (first, last, *windows)
        for got, want in zip(tensors, expected):
            torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
        assert tuple(x.data_ptr() for x in tensors) == pointers
