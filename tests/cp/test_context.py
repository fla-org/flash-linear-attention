# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import pytest
import torch

from fla.ops.cp import context
from fla.ops.cp.context import FLACPContext, build_cp_context, get_cp_cu_seqlens


@pytest.mark.parametrize(
    ('total_tokens', 'world_size'),
    [
        pytest.param(10, 3, id='remainder'),
        pytest.param(14, 4, id='remainder-varlen'),
    ],
)
@pytest.mark.parametrize('layout', ['contiguous', 'zigzag'])
def test_get_cp_cu_seqlens_invalid_total(total_tokens: int, world_size: int, layout: str):
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.long)
    divisor = r'2 \* world_size' if layout == 'zigzag' else 'world_size'
    num_parts = 2 * world_size if layout == 'zigzag' else world_size

    with pytest.raises(
        ValueError,
        match=rf"`total_tokens` \({total_tokens}\) must be divisible by `{divisor}` \({num_parts}\)",
    ):
        get_cp_cu_seqlens(cu_seqlens, world_size=world_size, rank=0, layout=layout)


@pytest.mark.parametrize('layout', ['contiguous', 'zigzag'])
def test_get_cp_cu_seqlens_empty_partition(layout: str):
    total_tokens = 2
    world_size = 3
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.long)
    divisor = r'2 \* world_size' if layout == 'zigzag' else 'world_size'
    num_parts = 2 * world_size if layout == 'zigzag' else world_size

    with pytest.raises(
        ValueError,
        match=rf"`total_tokens` \({total_tokens}\) must be at least `{divisor}` \({num_parts}\)",
    ):
        get_cp_cu_seqlens(cu_seqlens, world_size=world_size, rank=0, layout=layout)


def test_get_cp_cu_seqlens_preserves_divisible_varlen_partition():
    context = get_cp_cu_seqlens(
        torch.tensor([0, 5, 12], dtype=torch.long),
        world_size=3,
        rank=1,
    )

    assert context.cu_seqlens_cpu.tolist() == [0, 1, 4]
    assert context.pre_num_conv_tokens == 4
    assert context.pre_num_ranks == 1
    assert not context.is_first_rank
    assert context.post_num_ranks == 1
    assert not context.is_last_rank


@pytest.mark.parametrize('layout', ['contiguous', 'zigzag'])
def test_copy_for_backward(layout: str):
    context = get_cp_cu_seqlens(
        cu_seqlens=torch.tensor([0, 3, 9, 13, 16]),
        world_size=2,
        rank=0,
        conv1d_kernel_size=4,
        layout=layout,
    )
    context.pre_num_ranks_dev = torch.tensor([1], dtype=torch.int32)
    context.post_num_ranks_dev = torch.tensor([2], dtype=torch.int32)
    saved = context.copy_for_backward()

    assert saved is not context
    for name, value in vars(context).items():
        actual = getattr(saved, name)
        if name in ('cu_seqlens', 'cu_seqlens_cpu'):
            assert torch.equal(actual, value)
            assert actual.data_ptr() != value.data_ptr()
        elif isinstance(value, torch.Tensor):
            assert actual is value
        else:
            assert actual == value

    expected = saved.cu_seqlens.clone()
    context.cu_seqlens.zero_()
    context.cu_seqlens_cpu.zero_()
    assert torch.equal(saved.cu_seqlens, expected)
    assert torch.equal(saved.cu_seqlens_cpu, expected)


@pytest.mark.parametrize('entry', [FLACPContext, get_cp_cu_seqlens, build_cp_context], ids=['context', 'partition', 'build'])
@pytest.mark.parametrize('mode', ['omitted', 'true', 'false', 'unknown'])
def test_cp_precision_keyword(entry, mode, monkeypatch):
    monkeypatch.setattr(context.dist, 'get_world_size', lambda group: 2)
    monkeypatch.setattr(context.dist, 'get_rank', lambda group: 0)
    kwargs = dict(cu_seqlens=torch.tensor([0, 256]), group=object())
    if mode in ('true', 'false'):
        kwargs['use_tf32x3_affine_chain'] = mode == 'true'
    elif mode == 'unknown':
        kwargs['unknown_option'] = True
    if mode == 'false':
        with pytest.raises(ValueError, match='use_tf32x3_affine_chain=False'):
            entry(**kwargs)
    elif mode == 'unknown':
        with pytest.raises(TypeError, match='unknown_option'):
            entry(**kwargs)
    else:
        cp_context = entry(**kwargs)
        assert not hasattr(cp_context, 'use_tf32x3_affine_chain')
        copied = cp_context.copy_for_backward()
        assert copied.group is cp_context.group
        torch.testing.assert_close(copied.cu_seqlens, cp_context.cu_seqlens)


@pytest.mark.parametrize(
    ('is_nvidia', 'device_name', 'capability', 'supported'),
    [
        pytest.param(True, 'cuda:1', (7, 0), False, id='volta'),
        pytest.param(True, 'cuda:1', (7, 5), False, id='turing'),
        pytest.param(True, 'cuda:1', (8, 0), True, id='ampere'),
        pytest.param(True, 'cuda:1', (9, 0), True, id='hopper'),
        pytest.param(True, 'cuda:1', (10, 0), True, id='blackwell'),
        pytest.param(False, 'cuda:1', (9, 0), False, id='amd'),
        pytest.param(False, 'cpu', (0, 0), False, id='cpu'),
    ],
)
def test_cp_device_requirement(is_nvidia, device_name, capability, supported, monkeypatch):
    monkeypatch.setattr(context, 'IS_NVIDIA', is_nvidia)
    checked_devices = []

    def get_capability(device_index):
        checked_devices.append(device_index)
        return capability

    monkeypatch.setattr(context, 'get_device_capability', get_capability)
    if supported:
        context._check_cp_device(torch.device(device_name))
    else:
        with pytest.raises(RuntimeError, match='compute capability >= 8.0'):
            context._check_cp_device(torch.device(device_name))
    assert checked_devices == ([1] if is_nvidia else [])


@pytest.mark.parametrize('direction', ['fwd', 'bwd'])
def test_cp_pre_process_rejects_unsupported_device(direction, monkeypatch):
    from fla.ops.cp.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        chunk_gated_delta_rule_fwd_h_pre_process,
    )

    monkeypatch.setattr(context, 'IS_NVIDIA', False)
    x = torch.empty(1, 64, 1, 32)
    cp_context = FLACPContext(group=object())
    with pytest.raises(RuntimeError, match='compute capability >= 8.0'):
        if direction == 'fwd':
            chunk_gated_delta_rule_fwd_h_pre_process(k=x, w=x, u=x, context=cp_context)
        else:
            chunk_gated_delta_rule_bwd_dhu_pre_process(q=x, k=x, w=x, do=x, dv=x, context=cp_context)
