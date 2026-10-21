# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from dataclasses import fields

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


@pytest.mark.parametrize('entry', [FLACPContext, get_cp_cu_seqlens, build_cp_context], ids=['context', 'partition', 'build'])
@pytest.mark.parametrize('mode', ['omitted', 'true', 'false', 'unknown'])
def test_precision_keyword(entry, mode, monkeypatch):
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


@pytest.mark.parametrize('layout', ['contiguous', 'zigzag'])
def test_copy_for_backward(layout):
    cp_context = get_cp_cu_seqlens(
        cu_seqlens=torch.tensor([0, 5, 12]),
        world_size=2,
        rank=1,
        conv1d_kernel_size=4,
        layout=layout,
    )
    cp_context.group = object()
    cp_context.pre_num_ranks_dev = torch.tensor(1, dtype=torch.int32)
    cp_context.post_num_ranks_dev = torch.tensor(0, dtype=torch.int32)
    copied = cp_context.copy_for_backward()

    # Backward owns its sequence boundaries; graph replay must keep the same persistent rank-count tensors.
    for field in fields(cp_context):
        original = getattr(cp_context, field.name)
        saved = getattr(copied, field.name)
        if field.name in ('cu_seqlens', 'cu_seqlens_cpu'):
            torch.testing.assert_close(saved, original)
            assert saved.data_ptr() != original.data_ptr()
            saved.add_(1)
            torch.testing.assert_close(saved, original + 1)
        else:
            assert saved is original


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
def test_device_requirement(is_nvidia, device_name, capability, supported, monkeypatch):
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
def test_pre_process_unsupported_device(direction, monkeypatch):
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
