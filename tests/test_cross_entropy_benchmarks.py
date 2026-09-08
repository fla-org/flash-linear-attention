# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from benchmarks.ops.registry import generate_inputs, get_op, list_ops
from scripts.run_benchmark_compare import find_affected_op_names


@pytest.mark.parametrize('name', ['cross_entropy_loss', 'fused_linear_cross_entropy_loss'])
def test_cross_entropy_benchmark_registration(name):
    assert name in list_ops()
    config = get_op(name)
    assert len(config.default_shapes) >= 10
    assert {65535, 65536, 65537} <= {shape['D'] for shape in config.default_shapes.values()}
    assert config.test_file == 'tests/modules/test_cross_entropy.py'
    inputs = generate_inputs(config, B=1, T=63, H=8, D=67, dtype=torch.float32, device='cpu')
    target = inputs['target']
    assert target.shape == (63,)
    assert target[-1] == 66
    assert (target == -100).any()
    assert ((target == -100) | ((target >= 0) & (target < 67))).all()
    if name == 'cross_entropy_loss':
        assert inputs['logits'].shape == (63, 67)
    else:
        assert inputs['x'].shape == (63, 8)
        assert inputs['weight'].shape == (67, 8)


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        ('fla/modules/fused_cross_entropy.py', {'cross_entropy_loss', 'fused_linear_cross_entropy_loss'}),
        ('fla/modules/fused_linear_cross_entropy.py', {'fused_linear_cross_entropy_loss'}),
        ('fla/modules/backends/triton_ascend/__init__.py', {'cross_entropy_loss', 'fused_linear_cross_entropy_loss'}),
        ('tests/modules/test_cross_entropy.py', {'cross_entropy_loss', 'fused_linear_cross_entropy_loss'}),
        ('README.md', set()),
    ],
)
def test_cross_entropy_benchmark_discovery(path, expected):
    assert set(find_affected_op_names([path])) == expected


def test_cross_entropy_benchmark_preserves_ops_discovery():
    affected = find_affected_op_names(['fla/ops/kda/chunk.py'])
    assert 'chunk_kda' in affected
    assert not set(affected) & {'cross_entropy_loss', 'fused_linear_cross_entropy_loss'}
