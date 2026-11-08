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
from transformers.modeling_layers import GradientCheckpointingLayer

from fla.layers.gla import GatedLinearAttention
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.gla.configuration_gla import GLAConfig
from fla.models.modeling_utils import (
    LEGACY_RESIDUAL_KEY_MAPPING,
    FLAForCausalLM,
    FLAModel,
    FLAPreTrainedModel,
    get_attention,
)
from fla.modules import GatedMLP as GLAMLP
from fla.modules.residuals import BaseResidual

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache
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

        self.attn_residual = BaseResidual.from_config(config)
        self.mlp_residual = BaseResidual.from_config(config)

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
        hidden_states = state.hidden_states
        return hidden_states, attentions, past_key_values, state.with_hidden_states(hidden_states)


class GLAPreTrainedModel(FLAPreTrainedModel):

    config_class = GLAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GLABlock']
    _supports_cache_class = True
    _residual_block_types = (GLABlock,)
    _residual_key_mapping = LEGACY_RESIDUAL_KEY_MAPPING


class GLAModel(GLAPreTrainedModel, FLAModel):

    def __init__(self, config: GLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            GLABlock(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.output_residual = BaseResidual.from_config(config)

        self.gradient_checkpointing = False

        self.post_init()


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
