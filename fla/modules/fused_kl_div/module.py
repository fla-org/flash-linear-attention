# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn as nn

from fla.modules.fused_kl_div.ops import fused_kl_div_loss


class FusedKLDivLoss(nn.Module):

    def __init__(
        self,
        reduction: str = 'batchmean',
        accumulate_grad_in_fp32: bool = True,
    ):
        """
        Args:
            reduction (`str`):
                Specifies the reduction to apply to the output: 'batchmean'. Default: 'batchmean'.
            accumulate_grad_in_fp32 (`bool`):
                Whether to accumulate the student weight gradient in fp32 before casting it back
                to `weight.dtype`. Default: `True`.
        Note:
            FusedKLDivLoss only computes gradients for `x` and `weight`; `target_x` and
            `target_weight` are treated as frozen teacher tensors and must not require gradients.
        """
        super().__init__()

        assert reduction in ['batchmean'], f"reduction: {reduction} is not supported"

        self.reduction = reduction
        self.accumulate_grad_in_fp32 = accumulate_grad_in_fp32

    def forward(
        self,
        x: torch.Tensor,
        target_x: torch.Tensor,
        weight: torch.Tensor,
        target_weight: torch.Tensor,
    ):
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[batch_size * seq_len, hidden_size]`.
            target_x (`torch.Tensor`):
                Frozen teacher input tensor of shape `[batch_size * seq_len, hidden_size]`.
                Must not require gradients.
            weight (`torch.Tensor`):
                Tensor of shape `[vocab_size, hidden_size]`.
            target_weight (`torch.Tensor`):
                Frozen teacher weight tensor of shape `[vocab_size, hidden_size]`.
                Must not require gradients.
        Returns:
            loss
        """
        loss = fused_kl_div_loss(
            x=x,
            target_x=target_x,
            weight=weight,
            target_weight=target_weight,
            reduction=self.reduction,
            accumulate_grad_in_fp32=self.accumulate_grad_in_fp32,
        )
        return loss
