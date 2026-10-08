# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch

from fla.backends import BaseBackend, register_backend


@register_backend('modules.fused_linear_cross_entropy')
class TritonAscendBackend(BaseBackend):
    """Ascend NPU backend using triton-ascend kernels."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def logsumexp_fwd(
        self,
        x,
        scale=None,
        softcapping=None,
        dtype=None,
    ):
        from fla.modules.fused_linear_cross_entropy.backends.triton_ascend.ops import (
            logsumexp_fwd_npu,
        )
        return logsumexp_fwd_npu(x=x, scale=scale, softcapping=softcapping, dtype=dtype)

    def fused_linear_cross_entropy_fwd(
        self,
        x,
        target,
        weight,
        bias=None,
        ignore_index=-100,
        label_smoothing=0.0,
        logit_scale=1.0,
        logit_softcapping=None,
        num_chunks=8,
        reduction="mean",
        use_l2warp=False,
        l2_penalty_factor=1e-4,
        accumulate_grad_in_fp32=True,
        process_group=None,
    ):
        if process_group is not None and torch.distributed.get_world_size(process_group) > 1:
            raise NotImplementedError("Vocabulary-parallel fused linear cross entropy is not supported by the Ascend backend")
        from fla.modules.fused_linear_cross_entropy.backends.triton_ascend.ops import (
            fused_linear_cross_entropy_forward_npu,
        )
        return fused_linear_cross_entropy_forward_npu(
            x=x,
            target=target,
            weight=weight,
            bias=bias,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            logit_softcapping=logit_softcapping,
            num_chunks=num_chunks,
            reduction=reduction,
            use_l2warp=use_l2warp,
            l2_penalty_factor=l2_penalty_factor,
            accumulate_grad_in_fp32=accumulate_grad_in_fp32,
        )

    def fused_linear_cross_entropy_bwd(
        self,
        do,
        dx,
        dw,
        db,
    ):
        from fla.modules.fused_linear_cross_entropy.backends.triton_ascend.ops import (
            fused_linear_cross_entropy_backward_npu,
        )
        return fused_linear_cross_entropy_backward_npu(do=do, dx=dx, dw=dw, db=db)

    fused_linear_cross_entropy_forward = fused_linear_cross_entropy_fwd
    fused_linear_cross_entropy_backward = fused_linear_cross_entropy_bwd
