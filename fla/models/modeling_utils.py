# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.deprecation import deprecate_kwarg
from typing_extensions import Unpack

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp

if TYPE_CHECKING:
    from fla.models.cache_utils import Cache


class FLAPreTrainedModel(PreTrainedModel):
    """Shared Transformers integration without changing model parameter paths."""

    _is_stateful = True
    _input_embed_layer = "embeddings"

    def set_output_embeddings(self, new_embeddings):
        # allow reattaching a head after set_output_embeddings(None)
        self.lm_head = new_embeddings

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


class FLAForCausalLM(FLAPreTrainedModel):

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor | None = 0,
        num_items_in_batch: int | torch.Tensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        loss, logits = self._compute_logits_and_loss(
            hidden_states=hidden_states,
            labels=labels,
            logits_to_keep=logits_to_keep,
            num_items_in_batch=num_items_in_batch,
        )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
