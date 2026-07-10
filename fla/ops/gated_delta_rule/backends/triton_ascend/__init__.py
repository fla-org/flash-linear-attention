# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend (Huawei NPU) backend for gated_delta_rule."""
import torch
from fla.ops.backends import BaseBackend


class TritonAscendOpsBackend(BaseBackend):
    """Ascend NPU backend using triton-ascend kernels for ops-level dispatch."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def recompute_w_u_fwd_verifier(
        self,
        k,
        v,
        beta,
        g,
        A,
        cu_seqlens=None,
        chunk_indices=None,
    ) -> tuple[bool, str | None]:
        from fla.utils import IS_NPU
        if not IS_NPU:
            return False, "not running on NPU"
        if k.device.type != "npu":
            return False, "input device is not NPU"
        if all(t.dtype in (torch.float32, torch.float16, torch.bfloat16)
               for t in (k, v, beta, A)):
            return True, None
        return False, "unsupported dtype for NPU recompute_w_u_fwd"

    def recompute_w_u_fwd(
        self,
        k,
        v,
        beta,
        g,
        A,
        cu_seqlens=None,
        chunk_indices=None,
    ):
        import math
        from fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast import recompute_w_u_fwd_npu
        g_cumsum = g * math.log(2.0) if g is not None else None
        return recompute_w_u_fwd_npu(k, v, beta, g_cumsum, A, cu_seqlens)
