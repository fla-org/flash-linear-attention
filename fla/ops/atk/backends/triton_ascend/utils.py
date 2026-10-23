# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

def chunk_atk_fwd_verifier(*args, **kwargs) -> tuple[bool, str | None]:
    from fla.utils import IS_NPU
    k = args[0] if args else kwargs.get('k')
    if not IS_NPU:
        return False, "not running on NPU"
    if k is None or getattr(k, 'device', None) is None or k.device.type != "npu":
        return False, "input device is not NPU"
    return True, None
