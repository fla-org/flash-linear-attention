# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from types import SimpleNamespace

import pytest
import torch

from fla.ops.cp import context
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    chunk_gated_delta_rule_fwd_h_pre_process,
)
from fla.ops.cp.context import FLACPContext, build_cp_context, get_cp_cu_seqlens


@pytest.mark.parametrize(
    ('total_tokens', 'world_size', 'match'),
    [
        pytest.param(10, 3, 'must be divisible by', id='remainder'),
        pytest.param(14, 4, 'must be divisible by', id='remainder-four-ranks'),
        pytest.param(2, 3, 'must be at least', id='empty-partition'),
    ],
)
@pytest.mark.parametrize('layout', ['contiguous', 'zigzag'])
def test_get_cp_cu_seqlens_invalid_total(total_tokens: int, world_size: int, match: str, layout: str):
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.long)
    with pytest.raises(ValueError, match=match):
        get_cp_cu_seqlens(cu_seqlens, world_size=world_size, rank=0, layout=layout)


def test_get_cp_cu_seqlens_varlen():
    context = get_cp_cu_seqlens(torch.tensor([0, 5, 12], dtype=torch.long), world_size=3, rank=1)

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
    context.group = object()
    context.pre_num_ranks_dev = torch.tensor([1], dtype=torch.int32)
    context.post_num_ranks_dev = torch.tensor([2], dtype=torch.int32)
    saved = context.copy_for_backward()

    assert saved is not context
    assert saved.group is context.group
    assert saved.layout == layout
    assert saved.front_num_seqs == context.front_num_seqs
    assert saved.pre_num_ranks_by_part == context.pre_num_ranks_by_part
    assert saved.post_num_ranks_by_part == context.post_num_ranks_by_part
    # graph replay refreshes these tensors in place for both contexts
    assert saved.pre_num_ranks_dev is context.pre_num_ranks_dev
    assert saved.post_num_ranks_dev is context.post_num_ranks_dev

    expected = context.cu_seqlens.clone()
    context.cu_seqlens.zero_()
    context.cu_seqlens_cpu.zero_()
    assert torch.equal(saved.cu_seqlens, expected)
    assert torch.equal(saved.cu_seqlens_cpu, expected)


@pytest.fixture
def cp_group(monkeypatch):
    monkeypatch.setattr(context.dist, 'get_world_size', lambda group: 2)
    monkeypatch.setattr(context.dist, 'get_rank', lambda group: 0)
    return object()


@pytest.mark.parametrize(
    ('entry', 'expected'),
    [
        pytest.param(FLACPContext, [0, 256], id='context'),
        pytest.param(get_cp_cu_seqlens, [0, 128], id='partition'),
        pytest.param(build_cp_context, [0, 128], id='build'),
    ],
)
@pytest.mark.parametrize('kwargs', [{}, {'use_tf32x3_affine_chain': True}], ids=['default', 'legacy-true'])
def test_context_precision_keyword(cp_group, entry, expected, kwargs):
    cp_context = entry(cu_seqlens=torch.tensor([0, 256]), group=cp_group, **kwargs)
    assert cp_context.group is cp_group
    assert cp_context.cu_seqlens.tolist() == expected


@pytest.mark.parametrize('entry', [FLACPContext, get_cp_cu_seqlens, build_cp_context], ids=['context', 'partition', 'build'])
@pytest.mark.parametrize(
    ('kwargs', 'error', 'match'),
    [
        pytest.param({'use_tf32x3_affine_chain': False}, ValueError, 'use_tf32x3_affine_chain=False', id='legacy-false'),
        pytest.param({'unknown_option': True}, TypeError, 'unknown_option', id='unknown-keyword'),
    ],
)
def test_context_invalid_keyword(entry, kwargs, error, match):
    with pytest.raises(error, match=match):
        entry(cu_seqlens=torch.tensor([0, 256]), group=object(), **kwargs)


def test_context_positional_precision(cp_group):
    cu_seqlens = torch.tensor([0, 256])
    direct = FLACPContext(cp_group, cu_seqlens, None, None, None, None, None, None, None, True)
    partition = get_cp_cu_seqlens(cu_seqlens, None, 2, 0, cp_group, None, True)
    built = build_cp_context(cu_seqlens, cp_group, None, None, True)
    assert direct.cu_seqlens.tolist() == [0, 256]
    assert partition.cu_seqlens.tolist() == built.cu_seqlens.tolist() == [0, 128]


def test_context_invalid_positional_precision():
    cu_seqlens, group = torch.tensor([0, 256]), object()
    with pytest.raises(ValueError, match='use_tf32x3_affine_chain=False'):
        FLACPContext(group, cu_seqlens, None, None, None, None, None, None, None, False)
    with pytest.raises(ValueError, match='use_tf32x3_affine_chain=False'):
        get_cp_cu_seqlens(cu_seqlens, None, 2, 0, group, None, False)
    with pytest.raises(ValueError, match='use_tf32x3_affine_chain=False'):
        build_cp_context(cu_seqlens, group, None, None, False)


@pytest.fixture
def nvidia_devices(monkeypatch):
    monkeypatch.setattr(context, 'IS_NVIDIA', True)
    monkeypatch.setattr(context, 'IS_NPU', False)
    monkeypatch.setattr(context, 'get_device_capability', lambda index: {0: (7, 5), 1: (8, 0)}[index])


def test_check_cp_device_cuda(nvidia_devices):
    context._check_cp_device(torch.device('cuda:1'))


def test_check_cp_device_pre_ampere(nvidia_devices):
    with pytest.raises(RuntimeError, match='compute capability >= 8.0'):
        context._check_cp_device(torch.device('cuda:0'))


@pytest.mark.parametrize('device_name', ['cpu', 'cuda:1'], ids=['cpu', 'amd'])
def test_check_cp_device_unsupported_platform(monkeypatch, device_name):
    monkeypatch.setattr(context, 'IS_NVIDIA', False)
    monkeypatch.setattr(context, 'IS_NPU', False)
    with pytest.raises(RuntimeError, match='compute capability >= 8.0'):
        context._check_cp_device(torch.device(device_name))


def test_check_cp_device_ascend(monkeypatch):
    monkeypatch.setattr(context, 'IS_NVIDIA', False)
    monkeypatch.setattr(context, 'IS_NPU', True)
    context._check_cp_device(SimpleNamespace(type='npu', index=1))


def test_pre_process_unsupported_device():
    x = torch.empty(1, 64, 1, 32)
    cp_context = FLACPContext(group=object())
    with pytest.raises(RuntimeError, match='compute capability >= 8.0'):
        chunk_gated_delta_rule_fwd_h_pre_process(k=x, w=x, u=x, context=cp_context)
    with pytest.raises(RuntimeError, match='compute capability >= 8.0'):
        chunk_gated_delta_rule_bwd_dhu_pre_process(q=x, k=x, w=x, do=x, dv=x, context=cp_context)
