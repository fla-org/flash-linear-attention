# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from fla.backends import dispatch
from fla.modules.fused_cross_entropy.triton import cross_entropy_bwd, cross_entropy_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


class FusedCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        logits,
        target,
        label_smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        logit_softcapping=None,
        ignore_index=-100,
        inplace_backward=False,
        process_group: ProcessGroup | None = None,
        *,
        fwd_fn=None,
        bwd_fn=None,
    ):
        fwd_fn = cross_entropy_fwd if fwd_fn is None else fwd_fn
        ctx.bwd_fn = cross_entropy_bwd if bwd_fn is None else bwd_fn
        loss, z_loss, lse, total_classes, class_start_idx = fwd_fn(
            logits=logits,
            target=target,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            lse_square_scale=lse_square_scale,
            logit_softcapping=logit_softcapping,
            ignore_index=ignore_index,
            process_group=process_group,
        )
        ctx.save_for_backward(logits, target, lse)
        ctx.mark_non_differentiable(z_loss)
        ctx.label_smoothing = label_smoothing
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.logit_softcapping = logit_softcapping
        ctx.ignore_index = ignore_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.inplace_backward = inplace_backward

        return loss, z_loss

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dloss, dz_loss):
        # z-loss is returned for logging and is already included in the differentiable loss
        del dz_loss

        logits, target, lse = ctx.saved_tensors
        dlogits = ctx.bwd_fn(
            logits=logits,
            target=target,
            lse=lse,
            dloss=dloss,
            label_smoothing=ctx.label_smoothing,
            logit_scale=ctx.logit_scale,
            lse_square_scale=ctx.lse_square_scale,
            logit_softcapping=ctx.logit_softcapping,
            ignore_index=ctx.ignore_index,
            total_classes=ctx.total_classes,
            class_start_idx=ctx.class_start_idx,
            inplace_backward=ctx.inplace_backward,
        )
        return dlogits, None, None, None, None, None, None, None, None


fused_cross_entropy_forward = cross_entropy_fwd


CrossEntropyLossFunction = FusedCrossEntropyFunction


@dispatch
def cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
    inplace_backward: bool = False,
    process_group: ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute unreduced cross entropy and its non-differentiable z-loss component.

    Args:
        logits (torch.Tensor):
            Input logits of shape `[N, V]`.
        target (torch.Tensor):
            Global target indices of shape `[N]`.
        label_smoothing (float, Optional):
            Uniform label smoothing coefficient. Default: 0.0.
        logit_scale (float, Optional):
            Scale applied before softcapping. Default: 1.0.
        lse_square_scale (float, Optional):
            Coefficient of the squared logsumexp regularizer. Default: 0.0.
        logit_softcapping (float, Optional):
            Softcap applied as `softcap * tanh(logits / softcap)`. Default: `None`.
        ignore_index (int, Optional):
            Target index excluded from loss and gradients. Default: -100.
        inplace_backward (bool, Optional):
            Whether to overwrite logits with their gradients. Default: `False`.
        process_group (ProcessGroup, Optional):
            Group sharding the vocabulary into equal contiguous partitions. Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            FP32 loss and z-loss tensors of shape `[N]`. The loss includes z-loss.
    """
    return FusedCrossEntropyFunction.apply(
        logits,
        target,
        label_smoothing,
        logit_scale,
        lse_square_scale,
        logit_softcapping,
        ignore_index,
        inplace_backward,
        process_group,
    )


class FusedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        lse_square_scale: float = 0.0,
        logit_softcapping: float | None = None,
        inplace_backward: bool = False,
        process_group: ProcessGroup | None = None,
        return_z_loss: bool = False,
    ):
        """Cross entropy with optional logit transforms and vocabulary parallelism.

        Args:
            ignore_index (int, Optional):
                Target index excluded from loss and gradients. Default: -100.
            reduction (str, Optional):
                Reduction over non-ignored targets: `mean`, `sum`, or `none`. Default: `mean`.
            label_smoothing (float, Optional):
                Uniform label smoothing coefficient. Default: 0.0.
            logit_scale (float, Optional):
                Scale applied before softcapping. Default: 1.0.
            lse_square_scale (float, Optional):
                Coefficient of the squared logsumexp regularizer. Default: 0.0.
            logit_softcapping (float, Optional):
                Softcap applied as `softcap * tanh(logits / softcap)`. Default: `None`.
            inplace_backward (bool, Optional):
                Whether to overwrite logits with their gradients. Default: `False`.
            process_group (ProcessGroup, Optional):
                Group sharding the vocabulary into equal contiguous partitions. Default: `None`.
            return_z_loss (bool, Optional):
                Whether to also return z-loss for logging, without its own backward. Default: `False`.
        """
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise NotImplementedError(f"Unsupported reduction: {reduction}")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.logit_softcapping = logit_softcapping
        self.inplace_backward = inplace_backward
        self.process_group = process_group
        self.return_z_loss = return_z_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Compute loss from `[N, V]` logits and `[N]` targets.

        Return FP32 scalars for `mean` or `sum`, and `[N]` tensors for `none`.
        When `return_z_loss=True`, also return the reduced z-loss component.
        """
        assert input.device.type in ('cuda', 'npu', 'xpu') and target.device.type in ('cuda', 'npu', 'xpu'), (
            "Only support CUDA/NPU/XPU tensors"
        )
        loss, z_loss = cross_entropy_loss(
            logits=input,
            target=target,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            lse_square_scale=self.lse_square_scale,
            logit_softcapping=self.logit_softcapping,
            ignore_index=self.ignore_index,
            inplace_backward=self.inplace_backward,
            process_group=self.process_group,
        )
        if self.reduction != 'none':
            loss = loss.sum()
            if self.return_z_loss:
                z_loss = z_loss.sum()
            if self.reduction == 'mean':
                total = (target != self.ignore_index).sum()
                loss = loss / total
                if self.return_z_loss:
                    z_loss = z_loss / total
        return (loss, z_loss) if self.return_z_loss else loss
