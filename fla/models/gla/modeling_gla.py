# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import re
import warnings
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast

from fla.layers.gla import GatedLinearAttention
from fla.models.cache_utils import Cache
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.gla.configuration_gla import GLAConfig
from fla.models.modeling_utils import FLAForCausalLM, FLAPreTrainedModel, get_attention, get_residual
from fla.modules import GatedMLP as GLAMLP
from fla.modules.residuals import BaseResidual
from fla.modules.residuals.attnres import AttentionResidualState

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.modules.residuals.base import StateT


def _checkpoint_residual_state(function, *args, checkpoint_func):
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


class GLABlock(GradientCheckpointingLayer):
    """Run attention and MLP branches; residual modules own their inputs and state updates."""

    def __init__(self, config: GLAConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn = get_attention(config=config, layer_idx=layer_idx)
        if self.attn is None:
            self.attn = GatedLinearAttention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                feature_map=config.feature_map,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                use_output_gate=config.use_output_gate,
                gate_fn=config.hidden_act,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                clamp_min=config.clamp_min,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
            )

        self.mlp = GLAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        self.residual_attn = get_residual(config, fuse_residual=config.fuse_norm)
        self.residual_mlp = get_residual(config)

    def forward(
        self,
        state: StateT,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[StateT, torch.Tensor | None, Cache | None]:
        hidden_states, context = self.residual_attn.pre(state)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        state = self.residual_attn.post(hidden_states, context)

        hidden_states, context = self.residual_mlp.pre(state)
        hidden_states = self.mlp(hidden_states, **kwargs)
        state = self.residual_mlp.post(hidden_states, context)
        return state, attentions, past_key_values


class GLAPreTrainedModel(FLAPreTrainedModel):

    config_class = GLAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GLABlock']
    _supports_cache_class = True

    _checkpoint_conversion_mapping = {
        r'layers\.(\d+)\.attn_norm\.': r'layers.\1.residual_attn.norm.',
        r'layers\.(\d+)\.attn_res_proj\.': r'layers.\1.residual_attn.query.',
        r'layers\.(\d+)\.attn_res_norm\.': r'layers.\1.residual_attn.attnres_norm.',
        r'layers\.(\d+)\.mlp_norm\.': r'layers.\1.residual_mlp.norm.',
        r'layers\.(\d+)\.mlp_res_proj\.': r'layers.\1.residual_mlp.query.',
        r'layers\.(\d+)\.mlp_res_norm\.': r'layers.\1.residual_mlp.attnres_norm.',
        r'^(model\.|)norm\.': r'\1residual_output.norm.',
        r'^(model\.|)res_proj\.': r'\1residual_output.query.',
        r'^(model\.|)res_norm\.': r'\1residual_output.attnres_norm.',
    }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if cls._checkpoint_conversion_mapping:
            # transformers loads meta parameters without invoking module load hooks.
            kwargs['key_mapping'] = dict(kwargs.get('key_mapping') or {})
            for source, target in cls._checkpoint_conversion_mapping.items():
                kwargs['key_mapping'].setdefault(source, target)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self._checkpoint_conversion_mapping:
            for key in list(state_dict):
                if not key.startswith(prefix):
                    continue
                converted = key[len(prefix):]
                for pattern, replacement in self._checkpoint_conversion_mapping.items():
                    converted = re.sub(pattern, replacement, converted)
                converted = prefix + converted
                if converted != key:
                    state_dict.setdefault(converted, state_dict.pop(key))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        residual = getattr(self.base_model, 'residual_output', None)
        if not isinstance(residual, BaseResidual):
            return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)

        checkpoint_kwargs = {'use_reentrant': False, **(gradient_checkpointing_kwargs or {})}
        if checkpoint_kwargs['use_reentrant'] is None:
            checkpoint_kwargs['use_reentrant'] = False
        if checkpoint_kwargs['use_reentrant'] and not residual.supports_reentrant_checkpointing:
            raise ValueError(f'{type(residual).__name__} requires gradient checkpointing with use_reentrant=False')
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
        if checkpoint_kwargs['use_reentrant']:
            for module in self.modules():
                if isinstance(module, GradientCheckpointingLayer):
                    module._gradient_checkpointing_func = partial(
                        _checkpoint_residual_state,
                        checkpoint_func=module._gradient_checkpointing_func,
                    )


class GLAModel(GLAPreTrainedModel):

    def __init__(self, config: GLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GLABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.residual_output = get_residual(config)

        self.gradient_checkpointing = False

        self.post_init()

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

        state = self.residual_output.initialize(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                # preserve the existing hidden_states output: AttnRes exposes its current partial block sum.
                hidden_states = state.blocks[-1] if isinstance(state, AttentionResidualState) else state.residual
                all_hidden_states += (hidden_states,)
            state, attentions, past_key_values = layer(
                state,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.residual_output(state)
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


class GLAForCausalLM(GLAPreTrainedModel, FLAForCausalLM, FLAUnsupportedCacheGenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = GLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()
