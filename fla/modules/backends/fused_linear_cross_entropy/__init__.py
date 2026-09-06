# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration for fused_linear_cross_entropy."""

import torch

from fla.backends import BackendRegistry, BaseBackend


class TritonAscendFusedLinearCrossEntropyBackend(BaseBackend):
    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def logsumexp_fwd(self, x, scale: float | None = None, softcapping: float | None = None, dtype: torch.dtype | None = None):
        from fla.modules.backends.fused_linear_cross_entropy.triton_ascend import logsumexp_fwd
        return logsumexp_fwd(x=x, scale=scale, softcapping=softcapping, dtype=dtype)

    def fused_linear_cross_entropy_forward(
        self,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        logit_softcapping: float = None,
        num_chunks: int = 8,
        reduction: str = 'mean',
        use_l2warp: bool = False,
        l2_penalty_factor: float = 0.0001,
        accumulate_grad_in_fp32: bool = True,
    ):
        from fla.modules.backends.fused_linear_cross_entropy.triton_ascend import fused_linear_cross_entropy_forward
        return fused_linear_cross_entropy_forward(
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

    def fused_linear_cross_entropy_backward(self, do: torch.Tensor, dx: torch.Tensor, dw: torch.Tensor, db: torch.Tensor):
        from fla.modules.backends.fused_linear_cross_entropy.triton_ascend import fused_linear_cross_entropy_backward
        return fused_linear_cross_entropy_backward(do=do, dx=dx, dw=dw, db=db)


fused_linear_cross_entropy_registry = BackendRegistry("modules.fused_linear_cross_entropy")
fused_linear_cross_entropy_registry.register(TritonAscendFusedLinearCrossEntropyBackend())
