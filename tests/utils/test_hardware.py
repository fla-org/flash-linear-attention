# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

from fla.utils import hardware


def test_get_npu_target_arch_retries_per_device(monkeypatch, request):
    device_context = MagicMock(side_effect=nullcontext)
    get_target = MagicMock()
    monkeypatch.setattr(hardware.torch, 'npu', SimpleNamespace(device=device_context), raising=False)
    monkeypatch.setattr(hardware.triton.runtime, 'driver', SimpleNamespace(
        active=SimpleNamespace(get_current_target=get_target)))
    hardware._get_npu_target_arch.cache_clear()
    request.addfinalizer(hardware._get_npu_target_arch.cache_clear)
    get_target.side_effect = [
        RuntimeError('target temporarily unavailable'),
        SimpleNamespace(backend='npu', arch='Ascend910_9391'),
        SimpleNamespace(backend='npu', arch='Ascend950DT'),
        SimpleNamespace(backend='cuda', arch=90),
    ]
    assert hardware.get_npu_target_arch(0) == ''
    assert hardware.get_npu_target_arch(1) == 'Ascend910_9391'
    assert hardware.get_npu_target_arch(0) == 'Ascend950DT'
    assert hardware.get_npu_target_arch(2) == ''
    assert hardware.get_npu_target_arch(1) == 'Ascend910_9391'
    assert hardware.get_npu_target_arch(0) == 'Ascend950DT'
    assert [call.args[0] for call in device_context.call_args_list] == [0, 1, 0, 2]
    assert get_target.call_count == 4
