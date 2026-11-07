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
from fla.models.modeling_utils import FLAForCausalLM, FLAModel, FLAPreTrainedModel, get_attention
from fla.modules import GatedMLP as GLAMLP
from fla.modules.residuals import AttentionResidual, StandardResidual
from fla.modules.residuals.attnres import AttentionResidualState

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


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

        residual_kwargs = dict(hidden_size=config.hidden_size, norm_eps=config.norm_eps, fuse_norm=config.fuse_norm)
        if config.attnres_block_size is None:
            # fuse only attention -> MLP; materialize the residual sum at each layer boundary.
            self.residual_attn = StandardResidual(**residual_kwargs, fuse_residual=config.fuse_norm)
            self.residual_mlp = StandardResidual(**residual_kwargs)
        else:
            self.residual_attn = AttentionResidual(**residual_kwargs, block_size=config.attnres_block_size)
            self.residual_mlp = AttentionResidual(**residual_kwargs, block_size=config.attnres_block_size)

    def forward(
        self,
        state,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple:
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


class GLAModel(GLAPreTrainedModel, FLAModel):

    def __init__(self, config: GLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GLABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        residual_kwargs = dict(hidden_size=config.hidden_size, norm_eps=config.norm_eps, fuse_norm=config.fuse_norm)
        if config.attnres_block_size is None:
            self.residual_output = StandardResidual(**residual_kwargs)
        else:
            self.residual_output = AttentionResidual(**residual_kwargs, block_size=config.attnres_block_size)

        self.gradient_checkpointing = False

        self.post_init()

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
