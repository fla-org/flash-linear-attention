# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fla.utils import hardware


@pytest.mark.parametrize(
    ('target_arch', 'expected'),
    [
        ('NPU 910B4', 'a2/a3'),
        ('Ascend910_9391', 'a2/a3'),
        ('Ascend950DT', '950'),
        ('Ascend950PR', '950'),
        ('Ascend910_9599', '950'),
    ],
)
def test_get_npu_arch(monkeypatch, target_arch, expected):
    device_context = MagicMock(side_effect=lambda index: nullcontext())
    get_target = MagicMock(return_value=SimpleNamespace(backend='npu', arch=target_arch))
    monkeypatch.setattr(hardware.torch, 'npu', SimpleNamespace(device=device_context), raising=False)
    monkeypatch.setattr(hardware.triton.runtime, 'driver', SimpleNamespace(
        active=SimpleNamespace(get_current_target=get_target)))
    hardware.get_npu_arch.cache_clear()
    try:
        assert hardware.get_npu_arch(0) == expected
        assert hardware.get_npu_arch(0) == expected
        assert hardware.get_npu_arch(1) == expected
        assert get_target.call_count == 2
        assert [call.args[0] for call in device_context.call_args_list] == [0, 1]
    finally:
        hardware.get_npu_arch.cache_clear()
