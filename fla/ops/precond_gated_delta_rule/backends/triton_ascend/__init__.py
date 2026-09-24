# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for preconditioned gated_delta_rule ops."""

from __future__ import annotations

import torch

from fla.backends import BaseBackend, register


def _require_npu_device(*tensors: torch.Tensor | None) -> tuple[bool, str | None]:
    from fla.utils import IS_NPU
    if not IS_NPU:
        return False, "not running on NPU"
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.device.type != "npu":
            return False, "input device is not NPU"
    return True, None


@register('precond_gated_delta_rule')
class TritonAscendPrecondGDNBackend(BaseBackend):
    """Ascend NPU backend for preconditioned GDN chunk, fused-recurrent, KKT and WY kernels."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_precond_gated_delta_rule_verifier(
        self,
        q,
        k,
        v,
        g_atk,
        g,
        beta_atk,
        beta,
        **kwargs,
    ) -> tuple[bool, str | None]:
        return _require_npu_device(q, k, v, g_atk, g, beta_atk, beta)

    def fused_recurrent_precond_gated_delta_rule_fwd_verifier(
        self,
        q,
        k,
        v,
        g=None,
        gk=None,
        gv=None,
        beta=None,
        g_atk=None,
        beta_atk=None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        return _require_npu_device(q, k, v, g, gk, gv, beta, g_atk, beta_atk)

    def fused_recurrent_precond_gated_delta_rule_fwd(self, *args, **kwargs):
        from fla.ops.precond_gated_delta_rule.backends.triton_ascend.fused_recurrent import (
            fused_recurrent_precond_gated_delta_rule_fwd as fused_recurrent_precond_gated_delta_rule_fwd_npu,
        )
        return fused_recurrent_precond_gated_delta_rule_fwd_npu(*args, **kwargs)

    def chunk_precond_kkt_fwd_verifier(
        self,
        k,
        k_precond,
        g,
        beta,
        chunk_size=64,
        output_dtype=torch.float32,
        cu_seqlens=None,
    ) -> tuple[bool, str | None]:
        return _require_npu_device(k, k_precond, g, beta, cu_seqlens)

    def chunk_precond_kkt_fwd(self, *args, **kwargs):
        from fla.ops.precond_gated_delta_rule.backends.triton_ascend.chunk_precond_kkt_fwd import (
            chunk_precond_kkt_fwd_npu,
        )
        return chunk_precond_kkt_fwd_npu(*args, **kwargs)

    def prepare_precond_wy_repr_bwd_verifier(self, k, k_precond, v, *args, **kwargs):
        return _require_npu_device(k, k_precond, v)

    def prepare_precond_wy_repr_bwd(self, *args, **kwargs):
        from fla.ops.precond_gated_delta_rule.backends.triton_ascend.chunk_precond_wy_bwd import (
            prepare_precond_wy_repr_bwd_npu,
        )
        return prepare_precond_wy_repr_bwd_npu(*args, **kwargs)
