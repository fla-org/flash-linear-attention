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
from transformers.utils import logging

from fla.layers.gated_deltaproduct import GatedDeltaProduct
from fla.models.gated_deltaproduct.configuration_gated_deltaproduct import GatedDeltaProductConfig
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.modeling_utils import FLAForCausalLM, FLAModel, FLAPreTrainedModel, get_attention
from fla.modules import GatedMLP as GatedDeltaProductMLP
from fla.modules import RMSNorm
from fla.ops.attnres import fused_attnres

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


logger = logging.get_logger(__name__)


class GatedDeltaProductBlock(GradientCheckpointingLayer):

    def __init__(self, config: GatedDeltaProductConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = get_attention(config=config, layer_idx=layer_idx)
        if self.attn is None:
            self.attn = GatedDeltaProduct(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                use_output_gate=config.use_output_gate,
                use_forget_gate=config.use_forget_gate,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                allow_neg_eigval=config.allow_neg_eigval,
                num_householder=config.num_householder,
                layer_idx=layer_idx,
            )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedDeltaProductMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            self.attn_res_proj = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
            self.attn_res_norm = nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.norm_eps)
            self.mlp_res_proj = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
            self.mlp_res_norm = nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.norm_eps)
            block_size = config.attnres_block_size
            self.attnres_is_attn_boundary = (2 * layer_idx) % block_size == 0
            self.attnres_is_mlp_boundary = (2 * layer_idx + 1) % block_size == 0
            self.attn_res_proj._is_attnres_proj = True
            self.mlp_res_proj._is_attnres_proj = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        attnres_states: list[torch.Tensor] | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        if self.use_attnres:
            prefix_sum = hidden_states
            if attnres_states is None:
                # L=1 single-source: attnres is trivially identity (p=1, mix=v[0]);
                # apply the prenorm directly, matching the L>1 kernel path which
                # folds it via `output_rms_weight`. Mirrors Megatron-LM's bypass
                # at the first layer (where `block_residual` is empty).
                hidden_states = self.attn_norm(prefix_sum)
                attnres_states = [prefix_sum]
                prefix_sum = None
            else:
                residuals = [*attnres_states, prefix_sum]
                if self.attnres_is_attn_boundary:
                    attnres_states = residuals
                    prefix_sum = None
                hidden_states = fused_attnres(
                    query=self.attn_res_proj.weight,
                    residuals=residuals,
                    rms_weight=self.attn_res_norm.weight,
                    output_rms_weight=self.attn_norm.weight,
                    rms_eps=self.attn_res_norm.eps,
                )
        else:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        if self.use_attnres:
            prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
            residuals = [*attnres_states, prefix_sum]
            if self.attnres_is_mlp_boundary:
                attnres_states = residuals
                prefix_sum = None
            hidden_states = fused_attnres(
                query=self.mlp_res_proj.weight,
                residuals=residuals,
                rms_weight=self.mlp_res_norm.weight,
                output_rms_weight=self.mlp_norm.weight,
                rms_eps=self.mlp_res_norm.eps,
            )
        elif self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)

        if self.use_attnres:
            hidden_states = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        else:
            hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values, attnres_states)

        return outputs


class GatedDeltaProductPreTrainedModel(FLAPreTrainedModel):

    config_class = GatedDeltaProductConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GatedDeltaProductBlock']
    _supports_cache_class = True


class GatedDeltaProductModel(GatedDeltaProductPreTrainedModel, FLAModel):

    def __init__(self, config: GatedDeltaProductConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            GatedDeltaProductBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            self.res_proj = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
            self.res_norm = nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.norm_eps)
            self.res_proj._is_attnres_proj = True

        self.gradient_checkpointing = False

        self.post_init()


class GatedDeltaProductForCausalLM(GatedDeltaProductPreTrainedModel, FLAForCausalLM, FLAUnsupportedCacheGenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = GatedDeltaProductModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()
