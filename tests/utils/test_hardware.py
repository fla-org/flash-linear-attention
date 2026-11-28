# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fla.utils import hardware


@pytest.mark.parametrize(
    ('target_arch', 'expected'),
    [
        pytest.param('Ascend950DT_9582', True, id='950'),
        pytest.param('ascend910_9599', True, id='950-alias'),
        pytest.param('Ascend910_9382', False, id='a3'),
        pytest.param('gfx950', False, id='amd'),
        pytest.param(None, False, id='target-unavailable'),
    ],
)
def test_is_ascend_950(monkeypatch, request, target_arch, expected):
    query = Mock(return_value=SimpleNamespace(arch=target_arch))
    if target_arch is None:
        query.side_effect = RuntimeError('target unavailable')
    monkeypatch.setattr(hardware.triton.runtime, 'driver', SimpleNamespace(
        active=SimpleNamespace(get_current_target=query),
    ))
    hardware.get_device_arch.cache_clear()
    request.addfinalizer(hardware.get_device_arch.cache_clear)

    assert hardware.is_ascend_950() is expected
