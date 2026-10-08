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

from fla.modules.fused_cross_entropy.ops import cross_entropy_loss

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


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
