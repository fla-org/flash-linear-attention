# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
from copy import copy

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp


class FLAPreTrainedModel(PreTrainedModel):
    """Shared Transformers integration without changing model parameter paths."""

    _is_stateful = True

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if getattr(module, '_is_attnres_proj', False):
                nn.init.zeros_(module.weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            weight = None
            if hasattr(module, 'o_proj'):
                weight = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                weight = module.down_proj.weight
            if weight is not None:
                if prenorm_residual_strategy == 'rescale':
                    # reinitialize before GPT-2 residual scaling to avoid compounding repeated calls
                    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                    with torch.no_grad():
                        weight /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(weight)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")

    def _compute_logits_and_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor | None = 0,
        *,
        shift_labels: torch.LongTensor | None = None,
        num_items_in_batch: int | torch.Tensor | None = None,
        fuse_linear_cross_entropy: bool | None = None,
        use_l2warp: bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if fuse_linear_cross_entropy is None:
            fuse_linear_cross_entropy = getattr(self.config, "fuse_linear_cross_entropy", False)
        if use_l2warp is None:
            use_l2warp = getattr(self.config, "use_l2warp", False)
        has_labels = labels is not None or shift_labels is not None
        loss, logits = None, None
        if not fuse_linear_cross_entropy or not has_labels:
            if has_labels or logits_to_keep is None:
                selected_states = hidden_states
            elif isinstance(logits_to_keep, int):
                selected_states = hidden_states[:, -logits_to_keep:]
            else:
                selected_states = hidden_states[:, logits_to_keep]
            logits = self.lm_head(selected_states)
        if not has_labels:
            return loss, logits

        criterion = getattr(self, "criterion", None)
        reduction = "mean" if num_items_in_batch is None else "sum"
        if criterion is None:
            if fuse_linear_cross_entropy:
                criterion = FusedLinearCrossEntropyLoss(use_l2warp=use_l2warp, reduction=reduction)
            elif self.config.fuse_cross_entropy:
                criterion = FusedCrossEntropyLoss(inplace_backward=True, reduction=reduction)
            else:
                criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif num_items_in_batch is not None:
            if not hasattr(criterion, "reduction"):
                raise ValueError("A custom criterion must expose reduction to use num_items_in_batch.")
            criterion = copy(criterion)
            criterion.reduction = "sum"

        if shift_labels is None:
            labels = labels.to(hidden_states.device)
            shift_labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), dim=1)
        else:
            shift_labels = shift_labels.to(hidden_states.device)
        if fuse_linear_cross_entropy:
            loss = criterion(hidden_states, shift_labels, self.lm_head.weight, self.lm_head.bias)
        else:
            loss = criterion(logits.reshape(shift_labels.numel(), -1), shift_labels.reshape(-1))
            if use_l2warp:
                loss = l2_warp(loss, logits)
        if num_items_in_batch is not None:
            if isinstance(num_items_in_batch, torch.Tensor):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        return loss, logits
