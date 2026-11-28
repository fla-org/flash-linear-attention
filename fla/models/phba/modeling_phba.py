# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

from fla.layers.phba import PHBA
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.modeling_utils import FLAForCausalLM, FLAModel, FLAPreTrainedModel
from fla.models.phba.configuration_phba import PHBAConfig
from fla.modules import GatedMLP, RMSNorm
from fla.ops.phba.plans import _resolve_length_plan


class PHBABlock(GradientCheckpointingLayer):
    def __init__(self, config: PHBAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=config.elementwise_affine,
        )
        self.attn = PHBA(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            layer_idx=layer_idx,
            moba_chunk_size=config.moba_chunk_size,
            moba_topk=config.moba_topk,
            use_output_gate=config.use_output_gate,
            use_phba=config.use_phba,
            gsa_num_slots=config.gsa_num_slots,
            gate_logit_normalizer=config.gate_logit_normalizer,
            elementwise_affine=config.elementwise_affine,
        )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=config.elementwise_affine,
        )
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        attnres_states=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states, **kwargs)
        return hidden_states, attentions, past_key_values, attnres_states


class PHBAPreTrainedModel(FLAPreTrainedModel):
    config_class = PHBAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['PHBABlock']
    _supports_cache_class = False


class PHBAModel(PHBAPreTrainedModel, FLAModel):
    def __init__(self, config: PHBAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([PHBABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=config.elementwise_affine,
        )
        self.use_attnres = False
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if kwargs.get('use_cache', False) or kwargs.get('past_key_values') is not None:
            raise ValueError('PHBA supports cache-free prefill and training; set use_cache=False')
        inputs = input_ids if input_ids is not None else inputs_embeds
        if inputs is not None and attention_mask is None:
            cu_seqlens = kwargs.get('cu_seqlens')
            plan = _resolve_length_plan(
                inputs.shape[0], inputs.shape[1], inputs.device, cu_seqlens, kwargs.get('phba_length_plan'),
            )
            kwargs['phba_length_plan'] = plan
            if (
                cu_seqlens is not None and not plan._matches_source(cu_seqlens)
                or cu_seqlens is None and inputs.shape[0] == 1 and len(plan.lengths) > 1
            ):
                kwargs['cu_seqlens'] = plan.cu_seqlens
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


class PHBAForCausalLM(PHBAPreTrainedModel, FLAForCausalLM, FLAUnsupportedCacheGenerationMixin):
    _tied_weights_keys = {'lm_head.weight': 'model.embeddings.weight'}

    def __init__(self, config: PHBAConfig):
        super().__init__(config)
        self.model = PHBAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()
