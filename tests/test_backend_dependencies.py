# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest

from scripts.find_dependent_tests import find_backend_op_files


@pytest.mark.parametrize(
    ('changed', 'expected'),
    [
        ('fla/backends/registry.py', {'kda', 'dplr', 'norm'}),
        ('fla/ops/backends/__init__.py', {'kda', 'dplr', 'norm'}),
        ('fla/ops/kda/backends/__init__.py', {'kda'}),
        ('fla/ops/kda/backends/triton_ascend/chunk.py', {'kda'}),
        ('fla/ops/generalized_delta_rule/dplr/backends/__init__.py', {'dplr'}),
        ('fla/modules/backends/triton_ascend/layernorm.py', {'norm'}),
        ('fla/modules/backends/__init__.py', {'norm'}),
        ('fla/utils/_compat.py', set()),
    ],
)
def test_backend_changes_follow_dispatch_ownership(tmp_path, changed, expected):
    directory = tmp_path / 'fla'
    directory.mkdir()
    owners = {
        'kda': 'fla.ops.kda.backends',
        'dplr': 'fla.ops.generalized_delta_rule.dplr.backends',
        'norm': 'fla.modules.backends',
    }
    for name, owner in owners.items():
        (directory / f'{name}.py').write_text(f'from {owner} import dispatch\n@dispatch\ndef compute(x): return x\n')
    (directory / 'unrelated.py').write_text('def compute(x): return x\n')
    paths = find_backend_op_files([changed], tmp_path)
    assert set(paths) == {f'fla/{name}.py' for name in expected}
