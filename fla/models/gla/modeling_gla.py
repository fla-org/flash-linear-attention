# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast

from fla.layers.gla import GatedLinearAttention
from fla.models.cache_utils import Cache
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.gla.configuration_gla import GLAConfig
from fla.models.modeling_utils import FLAForCausalLM, FLAPreTrainedModel, get_attention
from fla.modules import GatedMLP as GLAMLP
from fla.modules.residuals import create_residual

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.modules.residuals.base import ResidualState


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

        residual_kwargs = dict(
            hidden_size=config.hidden_size,
            norm_eps=config.norm_eps,
            fuse_norm=config.fuse_norm,
            block_size=config.attnres_block_size,
        )
        self.attn_residual = create_residual(**residual_kwargs)
        self.mlp_residual = create_residual(**residual_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        residual_state: ResidualState | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple:
        if residual_state is None:
            state = self.attn_residual.initialize(hidden_states)
        else:
            state = residual_state.with_hidden_states(hidden_states)
        hidden_states, context = self.attn_residual.pre(state)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        state = self.attn_residual.post(hidden_states, context)

        hidden_states, context = self.mlp_residual.pre(state)
        hidden_states = self.mlp(hidden_states, **kwargs)
        state = self.mlp_residual.post(hidden_states, context)
        return state.hidden_states, attentions, past_key_values, state


class GLAPreTrainedModel(FLAPreTrainedModel):

    config_class = GLAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GLABlock']
    _supports_cache_class = True


class GLAModel(GLAPreTrainedModel):

    def __init__(self, config: GLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            GLABlock(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.output_residual = create_residual(
            hidden_size=config.hidden_size,
            norm_eps=config.norm_eps,
            fuse_norm=config.fuse_norm,
            block_size=config.attnres_block_size,
        )

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

        residual_state = self.output_residual.initialize(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attentions, past_key_values, residual_state = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                residual_state=residual_state,
                **kwargs,
            )

            if output_attentions:
                all_attns += (attentions,)

        residual_state = residual_state.with_hidden_states(hidden_states)
        hidden_states = self.output_residual(residual_state)
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
