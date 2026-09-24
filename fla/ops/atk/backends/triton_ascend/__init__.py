# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for ATK ops."""

from __future__ import annotations

from fla.ops.backends import BaseBackend


class TritonAscendATKBackend(BaseBackend):
    """Ascend NPU backend for chunked ATK forward and backward."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_atk_fwd_verifier(self, *args, **kwargs) -> tuple[bool, str | None]:
        from fla.utils import IS_NPU
        k = args[0] if args else kwargs.get('k')
        if not IS_NPU:
            return False, "not running on NPU"
        if k is None or getattr(k, 'device', None) is None or k.device.type != "npu":
            return False, "input device is not NPU"
        return True, None

    def chunk_atk_fwd(self, *args, **kwargs):
        from fla.ops.atk.backends.triton_ascend.chunk_atk_fwd import chunk_atk_fwd_npu
        return chunk_atk_fwd_npu(*args, **kwargs)

    def recompute_atk_fwd_verifier(self, *args, **kwargs) -> tuple[bool, str | None]:
        return self.chunk_atk_fwd_verifier(*args, **kwargs)

    def recompute_atk_fwd(self, *args, **kwargs):
        from fla.ops.atk.backends.triton_ascend.chunk_atk_fwd import recompute_atk_fwd_npu
        return recompute_atk_fwd_npu(*args, **kwargs)

    def chunk_atk_bwd_verifier(self, *args, **kwargs) -> tuple[bool, str | None]:
        return self.chunk_atk_fwd_verifier(*args, **kwargs)

    def chunk_atk_bwd(self, *args, **kwargs):
        from fla.ops.atk.backends.triton_ascend.chunk_atk_bwd import chunk_atk_bwd_npu
        return chunk_atk_bwd_npu(*args, **kwargs)
