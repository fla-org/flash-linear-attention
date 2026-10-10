# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


def simple_gla_verifier(q=None, *args, **kwargs) -> tuple[bool, str | None]:
    from fla.utils import IS_NPU
    if not IS_NPU:
        return False, "not running on NPU"
    if q is None or q.device.type != "npu":
        return False, "input device is not NPU"
    return True, None
