# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from transformers.modeling_layers import GradientCheckpointingLayer

from fla.layers.gla import GatedLinearAttention
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.gla.configuration_gla import GLAConfig
from fla.models.modeling_utils import FLAForCausalLM, FLAModel, FLAPreTrainedModel, get_attention
from fla.modules import GatedMLP as GLAMLP
from fla.modules.residuals import BaseResidual, get_residual_class

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


_RESIDUAL_KEY_MAPPING = {
    rf'layers\.(\d+)\.{sublayer}_{old}\.': rf'layers.\1.residual_{sublayer}.{new}.'
    for sublayer in ('attn', 'mlp')
    for old, new in (('norm', 'norm'), ('res_proj', 'query'), ('res_norm', 'key_norm'))
} | {
    r'^(model\.|)norm\.': r'\1residual_output.norm.',
    r'^(model\.|)res_proj\.': r'\1residual_output.query.',
    r'^(model\.|)res_norm\.': r'\1residual_output.key_norm.',
}


def _checkpoint_with_flattened_state(function, *args, checkpoint_func):
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

        residual_cls = get_residual_class(config.residual_mode)
        self.residual_attn = residual_cls(**config.get_residual_kwargs(sub_layer_idx=layer_idx * 2))
        self.residual_mlp = residual_cls(**config.get_residual_kwargs(sub_layer_idx=layer_idx * 2 + 1))

    def forward(
        self,
        state: Any,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[Any, torch.Tensor | None, Cache | None]:
        hidden_states, context = self.residual_attn.pre(state)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        state = self.residual_attn.post(state, hidden_states, context)

        hidden_states, context = self.residual_mlp.pre(state)
        hidden_states = self.mlp(hidden_states, **kwargs)
        state = self.residual_mlp.post(state, hidden_states, context)
        return state, attentions, past_key_values


class GLAPreTrainedModel(FLAPreTrainedModel):

    config_class = GLAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GLABlock']
    _supports_cache_class = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # transformers loads meta parameters without invoking module load hooks.
        kwargs['key_mapping'] = dict(kwargs.get('key_mapping') or {})
        for source, target in _RESIDUAL_KEY_MAPPING.items():
            kwargs['key_mapping'].setdefault(source, target)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        checkpoint_kwargs = {'use_reentrant': False, **(gradient_checkpointing_kwargs or {})}
        if checkpoint_kwargs['use_reentrant'] is None:
            checkpoint_kwargs['use_reentrant'] = False
        residual_cls = get_residual_class(self.config.residual_mode)
        if checkpoint_kwargs['use_reentrant'] and not residual_cls.supports_reentrant_checkpointing:
            raise ValueError(f'{residual_cls.__name__} requires gradient checkpointing with use_reentrant=False')
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
        if checkpoint_kwargs['use_reentrant']:
            for module in self.modules():
                if isinstance(module, GLABlock):
                    module._gradient_checkpointing_func = partial(
                        _checkpoint_with_flattened_state,
                        checkpoint_func=module._gradient_checkpointing_func,
                    )

    def _initialize_weights(self, module: nn.Module, *args, **kwargs):
        if getattr(module, '_is_residual_child', False):
            return
        if isinstance(module, BaseResidual):
            # a residual owns initialization of descendants even when its direct parameters were loaded.
            if not all(getattr(child, '_is_hf_initialized', False) for child in module.modules()):
                self._init_weights(module)
                for child in module.modules():
                    child._is_hf_initialized = True
            return
        super()._initialize_weights(module, *args, **kwargs)


class GLAModel(GLAPreTrainedModel, FLAModel):

    def __init__(self, config: GLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GLABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        residual_cls = get_residual_class(config.residual_mode)
        output_kwargs = config.get_residual_kwargs(sub_layer_idx=2 * config.num_hidden_layers, is_output=True)
        if not self.layers:
            raise ValueError('GLA requires at least one layer (num_hidden_layers >= 1)')
        self.residual_output = residual_cls(**output_kwargs)
        for residual in self.modules():
            if isinstance(residual, BaseResidual):
                for child in residual.modules():
                    if child is not residual:
                        child._is_residual_child = True

        self.gradient_checkpointing = False

        self.post_init()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in list(state_dict):
            if not key.startswith(prefix):
                continue
            converted = key[len(prefix):]
            for pattern, replacement in _RESIDUAL_KEY_MAPPING.items():
                converted = re.sub(pattern, replacement, converted)
            converted = prefix + converted
            if converted != key and converted not in state_dict:
                state_dict[converted] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _forward_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs: Unpack[dict],
    ) -> tuple:
        state = self.residual_output.initialize(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (self.residual_output.get_hidden_state(state),)
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

        hidden_states, _ = self.residual_output.pre(state)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, past_key_values, all_hidden_states, all_attns


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
