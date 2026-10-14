# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.cp.chunk_delta_h import compress_h0, expand_h0
from fla.ops.cp.context import FLACPContext, get_cp_cu_seqlens


@pytest.mark.parametrize(
    ('total_tokens', 'world_size'),
    [
        pytest.param(10, 3, id='remainder'),
        pytest.param(14, 4, id='remainder-varlen'),
    ],
)
def test_get_cp_cu_seqlens_rejects_non_divisible_total_tokens(total_tokens: int, world_size: int):
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match=rf"`total_tokens` \({total_tokens}\) must be divisible by `world_size` \({world_size}\)",
    ):
        get_cp_cu_seqlens(cu_seqlens, world_size=world_size, rank=0)


def test_get_cp_cu_seqlens_rejects_empty_rank_partition():
    total_tokens = 2
    world_size = 3
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match=rf"`total_tokens` \({total_tokens}\) must be at least `world_size` \({world_size}\)",
    ):
        get_cp_cu_seqlens(cu_seqlens, world_size=world_size, rank=0)


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
        use_tf32x3_affine_chain=True,
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


@pytest.mark.parametrize(
    ('layout', 'cu_seqlens', 'front_num_seqs', 'state_indices'),
    [
        pytest.param('contiguous', [0, 8], None, [0], id='contiguous-single'),
        pytest.param('contiguous', [0, 3, 8], None, [0], id='contiguous-varlen'),
        pytest.param('zigzag', [0, 4, 8], 1, [0, 1], id='zigzag-single'),
        pytest.param('zigzag', [0, 3, 4, 5, 8], 2, [0, 2], id='zigzag-varlen'),
    ],
)
@pytest.mark.parametrize('state_v_first', [False, True])
def test_compress_expand_h0(layout, cu_seqlens, front_num_seqs, state_indices, state_v_first):
    torch.manual_seed(42)
    context = FLACPContext(cu_seqlens=torch.tensor(cu_seqlens), layout=layout, front_num_seqs=front_num_seqs)
    state_shape = (2, 5, 3) if state_v_first else (2, 3, 5)
    h0 = torch.zeros(context.num_seqs, *state_shape)
    h0[state_indices] = torch.randn(len(state_indices), *state_shape)

    compressed = compress_h0(h0=h0, context=context)
    assert torch.equal(compressed, h0[state_indices])
    assert compressed.untyped_storage().nbytes() == compressed.numel() * compressed.element_size()
    assert torch.equal(expand_h0(h0=compressed, context=context), h0)
    assert compress_h0(h0=None, context=context) is None
    assert expand_h0(h0=None, context=context) is None
