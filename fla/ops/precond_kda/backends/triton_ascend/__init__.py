# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for Precond KDA ops."""

from __future__ import annotations

from fla.ops.backends import BaseBackend


def _verify_on_npu(*tensors) -> tuple[bool, str | None]:
    for t in tensors:
        if t is not None and getattr(t, 'device', None) is not None and t.device.type != "npu":
            return False, f"input device is not NPU: {t.device}"
    return True, None


class TritonAscendPrecondKDABackend(BaseBackend):
    """Ascend NPU backend for preconditioned KDA chunk and fused-recurrent ops."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_precond_kda_verifier(self, *args, **kwargs) -> tuple[bool, str | None]:
        q = args[0] if args else kwargs.get('q')
        k = args[1] if len(args) > 1 else kwargs.get('k')
        return _verify_on_npu(q, k)

    def chunk_precond_kda(self, *args, **kwargs):
        from fla.ops.precond_kda.backends.triton_ascend.chunk import chunk_precond_kda_npu
        return chunk_precond_kda_npu(*args, **kwargs)

    def fused_recurrent_precond_kda_verifier(self, *args, **kwargs) -> tuple[bool, str | None]:
        q = args[0] if args else kwargs.get('q')
        k = args[1] if len(args) > 1 else kwargs.get('k')
        return _verify_on_npu(q, k)

    def fused_recurrent_precond_kda(self, *args, **kwargs):
        from fla.ops.precond_kda.backends.triton_ascend.fused_recurrent import fused_recurrent_precond_kda_npu
        return fused_recurrent_precond_kda_npu(*args, **kwargs)

    def recompute_w_u_fwd_verifier(self, *args, **kwargs) -> tuple[bool, str | None]:
        k = args[0] if args else kwargs.get('k')
        return _verify_on_npu(k)

    def recompute_w_u_fwd(self, *args, **kwargs):
        from fla.ops.precond_kda.backends.triton_ascend.wy_fast import recompute_w_u_fwd_npu
        return recompute_w_u_fwd_npu(*args, **kwargs)
