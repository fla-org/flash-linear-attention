# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from types import SimpleNamespace

import pytest
import torch

from fla.ops.cp import chunk_bwd, chunk_fwd, context
from fla.ops.cp.context import build_cp_context, get_cp_cu_seqlens


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


@pytest.mark.parametrize(
    ('kernel', 'K', 'eight_warps'),
    [
        (chunk_fwd.chunk_cp_fwd_kernel_h, 128, True),
        (chunk_bwd.chunk_cp_bwd_kernel_dh, 128, True),
        (chunk_fwd.chunk_cp_fwd_kernel_h, 192, False),
        (chunk_bwd.chunk_cp_bwd_kernel_dh, 256, False),
    ],
    ids=['forward', 'backward', 'forward-wide', 'backward-wide'],
)
def test_summary_single_stage_configs(kernel, K, eight_warps):
    autotuner = kernel.fn
    configs = autotuner.configs
    if autotuner.early_config_prune is not None:
        configs = autotuner.early_config_prune(configs, {'K': K})
    # multi-stage K128 TF32x3 tiles can exceed shared memory and leave only spilling two-warp kernels
    assert any(config.num_stages == 1 and config.num_warps >= 4 for config in configs)
    assert any(config.num_stages == 1 and config.num_warps == 8 for config in configs) == eight_warps
