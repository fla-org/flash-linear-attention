# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for the momentum delta rule ops.

The community entries (``fla/ops/momentum_delta_rule/{chunk,fused_recurrent}.py``) are
kept pristine: they are decorated with ``@dispatch('momentum_delta_rule')``, and on NPU
the dispatch routes to the implementations below. Every other platform keeps running
the community torch reference.
"""

from __future__ import annotations

import torch

from fla.backends import BaseBackend, register


@register('momentum_delta_rule')
class TritonAscendMomentumDeltaRuleBackend(BaseBackend):
    """Ascend NPU backend for the momentum delta rule chunk/recurrent entries."""

    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_momentum_delta_rule_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        p: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        eta: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        use_p_times_alpha: bool = True,
        chunk_size: int = 64,
    ) -> tuple[bool, str | None]:
        if q.device.type != "npu":
            return False, "chunk_momentum_delta_rule NPU override requires NPU tensors"
        if cu_seqlens is not None:
            return False, "variable-length `cu_seqlens` not supported by the NPU override"
        return True, None

    def chunk_momentum_delta_rule(self, *args, **kwargs):
        from fla.ops.momentum_delta_rule.backends.triton_ascend.chunk_momentum_delta import chunk_momentum_delta_rule_npu
        return chunk_momentum_delta_rule_npu(*args, **kwargs)

    def fused_recurrent_momentum_delta_rule_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        p: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        eta: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        use_p_times_alpha: bool = True,
    ) -> tuple[bool, str | None]:
        if q.device.type != "npu":
            return False, "fused_recurrent_momentum_delta_rule NPU override requires NPU tensors"
        if cu_seqlens is not None:
            return False, "variable-length `cu_seqlens` not supported by the NPU override"
        return True, None

    def fused_recurrent_momentum_delta_rule(self, *args, **kwargs):
        from fla.ops.momentum_delta_rule.backends.triton_ascend.fused_recurrent import fused_recurrent_momentum_delta_rule_npu
        return fused_recurrent_momentum_delta_rule_npu(*args, **kwargs)


__all__ = ['TritonAscendMomentumDeltaRuleBackend']
