# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for AttnRes."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from fla.ops.backends import BaseBackend


class TritonAscendAttnResBackend(BaseBackend):
    """Ascend NPU backend for fused AttnRes."""

    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def fused_attnres_verifier(
        self,
        query: torch.Tensor,
        residuals: Sequence[torch.Tensor],
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor | None = None,
        rms_eps: float = 1e-6,
        scale: float = 1.0,
        return_weights: bool = False,
        checkpoint_level: int = 1,
        **kwargs,
    ) -> tuple[bool, str | None]:
        del query, rms_weight, output_rms_weight, rms_eps, scale, return_weights, kwargs
        if isinstance(residuals, torch.Tensor):
            if residuals.ndim != 4 or residuals.shape[0] == 0:
                return False, 'attnres requires at least one residual source'
        elif not residuals:
            return False, 'attnres requires at least one residual source'
        return True, None

    def fused_attnres(self, *args, **kwargs):
        from fla.ops.attnres.backends.triton_ascend.fused import fused_attnres_npu
        return fused_attnres_npu(*args, **kwargs)


__all__ = ['TritonAscendAttnResBackend']
