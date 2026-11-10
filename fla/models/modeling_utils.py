# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
import re
import warnings
from copy import copy
from functools import partial
from inspect import signature

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from transformers.configuration_utils import PretrainedConfig
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.deprecation import deprecate_kwarg
from typing_extensions import Unpack

from fla.layers.attn import Attention
from fla.models.cache_utils import Cache
from fla.models.configuration_utils import get_attention_spec
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp
from fla.ops.attnres import fused_attnres

LEGACY_RESIDUAL_KEY_MAPPING = {
    r'layers\.(\d+)\.attn_norm\.': r'layers.\1.attn_residual.norm.',
    r'layers\.(\d+)\.attn_res_proj\.': r'layers.\1.attn_residual.query.',
    r'layers\.(\d+)\.attn_res_norm\.': r'layers.\1.attn_residual.attnres_norm.',
    r'layers\.(\d+)\.mlp_norm\.': r'layers.\1.mlp_residual.norm.',
    r'layers\.(\d+)\.mlp_res_proj\.': r'layers.\1.mlp_residual.query.',
    r'layers\.(\d+)\.mlp_res_norm\.': r'layers.\1.mlp_residual.attnres_norm.',
    r'^(model\.|)norm\.': r'\1output_residual.norm.',
    r'^(model\.|)res_proj\.': r'\1output_residual.query.',
    r'^(model\.|)res_norm\.': r'\1output_residual.attnres_norm.',
}


def register_residual_key_mapping(model_type: str, key_mapping: dict[str, str]) -> None:
    """Register legacy residual parameter names for Transformers loading and saving."""
    if key_mapping and get_checkpoint_conversion_mapping(model_type) is None:
        register_checkpoint_conversion_mapping(model_type, [
            WeightRenaming(source_patterns=source, target_patterns=target) for source, target in key_mapping.items()
        ])


def get_attention(config: PretrainedConfig, layer_idx: int) -> Attention | None:
    """Create standard attention for a selected layer, or return None for a native mixer."""
    attn_spec = get_attention_spec(config.attn, layer_idx=layer_idx)
    if attn_spec is None:
        return None
    return Attention(
        hidden_size=config.hidden_size,
        num_heads=attn_spec['num_heads'],
        num_kv_heads=attn_spec['num_kv_heads'],
        qkv_bias=attn_spec['qkv_bias'],
        window_size=attn_spec['window_size'],
        rope_theta=attn_spec['rope_theta'],
        max_position_embeddings=config.max_position_embeddings,
        layer_idx=layer_idx,
    )


def checkpoint_residual_state(function, *args, checkpoint_func):
    """Expose tensors inside residual state to reentrant checkpointing."""
    flat_args, input_spec = tree_flatten(args)
    output_spec = None

    def forward(*flat_args):
        nonlocal output_spec
        outputs = function(*tree_unflatten(flat_args, input_spec))
        flat_outputs, output_spec = tree_flatten(outputs)
        return tuple(flat_outputs)

    outputs = checkpoint_func(forward, *flat_args)
    return tree_unflatten(outputs, output_spec)


class FLAPreTrainedModel(PreTrainedModel):
    """Shared Transformers integration without changing model parameter paths."""

    _is_stateful = True
    _input_embed_layer = "embeddings"
    _residual_block_types: tuple[type[nn.Module], ...] = ()
    _residual_key_mapping: dict[str, str] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._residual_key_mapping:
            register_residual_key_mapping(cls.config_class.model_type, cls._residual_key_mapping)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, *args, **kwargs):
        if not self._residual_block_types:
            return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs, *args, **kwargs)

        checkpoint_kwargs = {'use_reentrant': False, **(gradient_checkpointing_kwargs or {})}
        if checkpoint_kwargs['use_reentrant'] is None:
            checkpoint_kwargs['use_reentrant'] = False

        residual = self.base_model.output_residual
        if checkpoint_kwargs['use_reentrant'] and not residual.supports_reentrant_checkpointing:
            raise ValueError(f'{type(residual).__name__} requires gradient checkpointing with use_reentrant=False')

        super().gradient_checkpointing_enable(checkpoint_kwargs, *args, **kwargs)
        if checkpoint_kwargs['use_reentrant']:
            for module in self.modules():
                if isinstance(module, self._residual_block_types):
                    module._gradient_checkpointing_func = partial(
                        checkpoint_residual_state,
                        checkpoint_func=module._gradient_checkpointing_func,
                    )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self._residual_key_mapping:
            for name in list(state_dict):
                if not name.startswith(prefix):
                    continue
                converted = name[len(prefix):]
                for source, target in self._residual_key_mapping.items():
                    converted = re.sub(source, target, converted)
                converted = prefix + converted
                if converted != name:
                    state_dict.setdefault(converted, state_dict.pop(name))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        if self._residual_key_mapping and 'save_original_format' not in kwargs:
            # Keep new parameter names by default while honoring explicitly supplied positional save options.
            arguments = signature(super().save_pretrained).bind_partial(*args, **kwargs).arguments
            if 'save_original_format' not in arguments:
                kwargs['save_original_format'] = False
        return super().save_pretrained(*args, **kwargs)

    def set_output_embeddings(self, new_embeddings):
        # allow reattaching a head after set_output_embeddings(None)
        self.lm_head = new_embeddings

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


class FLAModel(FLAPreTrainedModel):

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            warnings.warn(f"`{self.__class__.__name__}` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if self._residual_block_types:
            residual_state = self.output_residual.initialize(hidden_states)

            all_hidden_states = () if output_hidden_states else None
            all_attns = () if output_attentions else None
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (residual_state.hidden_states,)

                residual_state, attentions, past_key_values = layer(
                    residual_state,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs,
                )

                if output_attentions:
                    all_attns += (attentions,)

            hidden_states = self.output_residual(residual_state)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        else:
            attnres_states: list[torch.Tensor] | None = None

            all_hidden_states = () if output_hidden_states else None
            all_attns = () if output_attentions else None
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                hidden_states, attentions, past_key_values, attnres_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    attnres_states=attnres_states,
                    **kwargs,
                )

                if output_attentions:
                    all_attns += (attentions,)

            if self.use_attnres:
                # top-level attnres aggregation; `self.norm` is folded into the
                # kernel via `output_rms_weight` so we don't double-norm.
                residuals = [*attnres_states, hidden_states]
                hidden_states = fused_attnres(
                    query=self.res_proj.weight,
                    residuals=residuals,
                    rms_weight=self.res_norm.weight,
                    output_rms_weight=self.norm.weight,
                    rms_eps=self.res_norm.eps,
                )
            else:
                hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


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
