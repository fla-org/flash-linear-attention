# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from pathlib import Path

import pytest

from scripts.find_dependent_tests import find_backend_op_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('filename', ['__init__.py', 'triton_ascend.py'])
@pytest.mark.parametrize('operation', ['layernorm', 'rotary', 'causal_conv1d'])
def test_module_backend_changes_select_the_corresponding_entry_point(operation, filename):
    changed = [f'fla/modules/backends/{operation}/{filename}']
    expected = 'fla/modules/conv/triton/ops.py' if operation == 'causal_conv1d' else f'fla/modules/{operation}.py'
    assert find_backend_op_files(changed, ROOT) == [expected]


def test_legacy_backend_changes_select_all_module_entry_points():
    selected = find_backend_op_files(['fla/modules/backends/triton_ascend/layernorm.py'], ROOT)
    assert 'fla/modules/layernorm.py' in selected
    assert 'fla/modules/conv/triton/ops.py' in selected
    assert 'fla/modules/rotary.py' in selected


def test_op_backend_changes_keep_their_existing_mapping():
    selected = find_backend_op_files(['fla/ops/gla/backends/triton_ascend/chunk.py'], ROOT)
    assert 'fla/ops/gla/chunk.py' in selected
