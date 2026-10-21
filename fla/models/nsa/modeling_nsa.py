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

from fla.layers.nsa import NativeSparseAttention
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.modeling_utils import FLAForCausalLM, FLAModel, FLAPreTrainedModel
from fla.models.nsa.configuration_nsa import NSAConfig
from fla.modules import GatedMLP as NSAMLP
from fla.modules import RMSNorm
from fla.ops.attnres import fused_attnres

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


logger = logging.get_logger(__name__)


class NSABlock(GradientCheckpointingLayer):

    def __init__(self, config: NSAConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = NativeSparseAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            qkv_bias=config.qkv_bias,
            block_size=config.block_size,
            block_counts=config.block_counts,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = NSAMLP(
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
            # a sub-layer "starts a new block" if its global index
            # (`2*layer_idx` for attn, `2*layer_idx+1` for mlp) is a multiple
            # of `attnres_block_size`. when `True`, the incoming `prefix_sum`
            # represents the previous block's complete sum (or the token
            # embedding for the very first sub-layer) and gets cat'd into
            # `attnres_states` before this sub-layer's attnres call.
            block_size = config.attnres_block_size
            self.attnres_is_attn_boundary = (2 * layer_idx) % block_size == 0
            self.attnres_is_mlp_boundary = (2 * layer_idx + 1) % block_size == 0
            # tag so `_init_weights` keeps the zero init (paper §5)
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
            # incoming `hidden_states` is the running `prefix_sum`
            # (= previous layer's `output = prefix_sum + mlp_out`)
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
                    # prev block's sum becomes a new residual entry
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
            # accumulate attn output into the running `prefix_sum`
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
            # returned `hidden_states` carries the running `prefix_sum` to
            # the next layer (single-tensor pp transmission, no separate carry)
            hidden_states = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        else:
            hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values, attnres_states)

        return outputs


class NSAPreTrainedModel(FLAPreTrainedModel):

    config_class = NSAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['NSABlock']
    _supports_cache_class = True


class NSAModel(NSAPreTrainedModel, FLAModel):

    def __init__(self, config: NSAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([NSABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            # top-level attnres aggregation params; `self.norm` still applies afterward
            self.res_proj = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
            self.res_norm = nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.norm_eps)
            self.res_proj._is_attnres_proj = True

        self.gradient_checkpointing = False

        self.post_init()


class NSAForCausalLM(NSAPreTrainedModel, FLAForCausalLM, FLAUnsupportedCacheGenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = NSAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()
