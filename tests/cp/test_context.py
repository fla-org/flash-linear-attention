# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import pytest
import torch

from fla.ops.cp.context import get_cp_cu_seqlens


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
