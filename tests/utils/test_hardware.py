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


@pytest.fixture
def npu_target_query(monkeypatch):
    query = Mock()
    monkeypatch.setattr(hardware.triton.runtime, 'driver', SimpleNamespace(
        active=SimpleNamespace(get_current_target=query),
    ))
    hardware._get_npu_target_arch.cache_clear()
    yield query
    hardware._get_npu_target_arch.cache_clear()


def test_get_npu_target_arch_retries_and_caches(npu_target_query, monkeypatch):
    query = npu_target_query
    current_device = Mock(return_value=0)
    monkeypatch.setattr(hardware.torch, 'npu', SimpleNamespace(current_device=current_device), raising=False)
    query.side_effect = [
        SystemError('target temporarily unavailable'),
        SimpleNamespace(backend='npu', arch='Ascend950DT_9582'),
    ]

    assert hardware.get_npu_target_arch() == 'Ascend950DT_9582'
    current_device.return_value = 1
    assert hardware.get_npu_target_arch() == 'Ascend950DT_9582'
    assert query.call_count == 2


@pytest.mark.parametrize(
    ('retry_result', 'retry_message'),
    [
        pytest.param(SystemError('retry failed'), 'retry failed', id='exception'),
        pytest.param(
            SimpleNamespace(backend='npu', arch=''),
            'Triton did not report an NPU target architecture.',
            id='invalid-target',
        ),
    ],
)
def test_get_npu_target_arch_preserves_query_error(npu_target_query, retry_result, retry_message):
    query = npu_target_query
    first_error = OSError('target lookup failed')
    query.side_effect = [
        first_error,
        retry_result,
        SimpleNamespace(backend='npu', arch='Ascend910_9391'),
    ]

    with pytest.raises(OSError, match='target lookup failed') as exc_info:
        hardware.get_npu_target_arch()
    assert exc_info.value is first_error
    assert str(exc_info.value.__cause__) == retry_message
    assert query.call_count == 2
    assert hardware.get_npu_target_arch() == 'Ascend910_9391'
    assert query.call_count == 3


@pytest.mark.parametrize(('backend', 'arch'), [('cuda', 90), ('npu', '')])
def test_get_npu_target_arch_rejects_invalid_target(npu_target_query, backend, arch):
    query = npu_target_query
    query.return_value = SimpleNamespace(backend=backend, arch=arch)

    with pytest.raises(RuntimeError, match='did not report an NPU target architecture'):
        hardware.get_npu_target_arch()
    assert query.call_count == 2
