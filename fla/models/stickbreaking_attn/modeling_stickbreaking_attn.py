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
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.stickbreaking_attn import StickBreakingAttention
from fla.models.stickbreaking_attn.configuration_stickbreaking_attn import StickBreakingAttentionConfig
from fla.models.transformer.modeling_transformer import TransformerForCausalLM, TransformerModel, TransformerPreTrainedModel
from fla.modules import GatedMLP, RMSNorm

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer


class StickBreakingAttentionBlock(GradientCheckpointingLayer):

    def __init__(self, config: StickBreakingAttentionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = StickBreakingAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            attend_current=config.attend_current,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
        )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
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
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
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
        return hidden_states, attentions, past_key_values, None


class StickBreakingAttentionModel(TransformerModel):
    """Pre-norm decoder using stick-breaking attention without KV caching."""

    config_class = StickBreakingAttentionConfig
    _no_split_modules = ['StickBreakingAttentionBlock']
    _supports_cache_class = False

    def __init__(self, config: StickBreakingAttentionConfig):
        # initialize pretrained-model machinery without constructing softmax attention blocks
        TransformerPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([StickBreakingAttentionBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.use_attnres = False
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs,
    ):
        if use_cache or past_key_values is not None or (use_cache is None and self.config.use_cache):
            raise NotImplementedError("Stick-breaking attention does not support KV-cache decoding. Use use_cache=False.")
        if output_attentions or (output_attentions is None and self.config.output_attentions):
            raise NotImplementedError("Stick-breaking attention does not return attention matrices.")
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            **kwargs,
        )


class StickBreakingAttentionForCausalLM(TransformerForCausalLM):
    """Causal language model with full-prefix generation and isolated packed-sequence labels."""

    config_class = StickBreakingAttentionConfig
    _tied_weights_keys = {'lm_head.weight': 'model.embeddings.weight'}
    _no_split_modules = ['StickBreakingAttentionBlock']
    _supports_cache_class = False

    def __init__(self, config: StickBreakingAttentionConfig):
        TransformerPreTrainedModel.__init__(self, config)
        self.model = StickBreakingAttentionModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    @deprecate_kwarg('num_logits_to_keep', version='4.50', new_name='logits_to_keep')
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ):
        if labels is not None and cu_seqlens is not None:
            ignore_index = self.criterion.ignore_index if self.criterion is not None else -100
            # the extra element also covers boundaries of trailing empty sequences
            labels = torch.cat((labels, torch.full_like(labels[:, :1], ignore_index)), dim=1)
            labels[:, cu_seqlens[1:-1]] = ignore_index
            labels = labels[:, :-1]
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
