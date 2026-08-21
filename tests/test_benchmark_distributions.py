# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from benchmarks.distributions import sample_lognormal_packed_lengths
from benchmarks.ops.registry import generate_inputs, get_op


def test_lognormal_lengths_are_seeded_bounded_and_exact():
    first = sample_lognormal_packed_lengths(
        total_tokens=8192,
        num_sequences=32,
        min_length=16,
        max_length=1024,
        sigma=1.0,
        generator=torch.Generator().manual_seed(42),
    )
    second = sample_lognormal_packed_lengths(
        total_tokens=8192,
        num_sequences=32,
        min_length=16,
        max_length=1024,
        sigma=1.0,
        generator=torch.Generator().manual_seed(42),
    )

    assert torch.equal(first, second)
    assert first.sum() == 8192
    assert first.min() >= 16
    assert first.max() <= 1024
    assert first.float().quantile(0.9) > 2 * first.float().median()


@pytest.mark.parametrize(
    'kwargs',
    [
        {'total_tokens': 0, 'num_sequences': 1, 'max_length': 16},
        {'total_tokens': 32, 'num_sequences': 0, 'max_length': 16},
        {'total_tokens': 32, 'num_sequences': 2, 'min_length': 17, 'max_length': 16},
        {'total_tokens': 33, 'num_sequences': 2, 'min_length': 16, 'max_length': 16},
    ],
)
def test_lognormal_lengths_reject_invalid_packing(kwargs):
    with pytest.raises(ValueError):
        sample_lognormal_packed_lengths(**kwargs)


def test_realistic_gla_gate_matches_layer_normalizer():
    config = get_op('chunk_gla')
    torch.manual_seed(42)
    synthetic = generate_inputs(config, 1, 64, 2, 16, dtype=torch.float32, device='cpu')
    torch.manual_seed(42)
    realistic = generate_inputs(
        config,
        1,
        64,
        2,
        16,
        dtype=torch.float32,
        device='cpu',
        input_profile='realistic',
    )

    assert realistic['g'].max() <= 0
    assert torch.allclose(realistic['g'], synthetic['g'] / 16)


@pytest.mark.parametrize('op_name', ['chunk_gdn', 'chunk_comba'])
def test_realistic_learned_decay_is_nonsymmetric(op_name):
    torch.manual_seed(42)
    inputs = generate_inputs(
        get_op(op_name),
        1,
        256,
        4,
        16,
        dtype=torch.float32,
        device='cpu',
        input_profile='realistic',
    )

    assert inputs['q'].mean() > 0
    assert inputs['g'].max() < 0
    assert inputs['g'].std() > 0

    if op_name == 'chunk_comba':
        expected_p = inputs['k'].detach() * torch.ones(4).sigmoid()[None, None, :, None]
        assert torch.allclose(inputs['p'], expected_p)


def test_realistic_kda_gate_respects_safe_bound():
    torch.manual_seed(42)
    inputs = generate_inputs(
        get_op('chunk_kda'),
        1,
        256,
        4,
        16,
        dtype=torch.float32,
        device='cpu',
        input_profile='realistic',
    )

    assert inputs['g'].min() >= -5
    assert inputs['g'].max() <= 0
    assert inputs['g'].min() < 0
    assert inputs['g'].std() > 0
