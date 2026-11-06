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

from fla.layers.mlstm import MLSTM
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.mlstm.configuration_mlstm import MLSTMConfig
from fla.models.modeling_utils import FLAForCausalLM, FLAModel, FLAPreTrainedModel, get_attention
from fla.modules import GatedMLP, RMSNorm, ShortConvolution

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


class MLSTMBlock(GradientCheckpointingLayer):

    def __init__(self, config: MLSTMConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = get_attention(config=config, layer_idx=layer_idx)
        if self.attn is None:
            self.attn = MLSTM(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                proj_factor=config.proj_factor,
                qkv_proj_blocksize=config.qkv_proj_blocksize,
                conv_size=config.conv_size,
                conv_bias=config.conv_bias,
                bias=config.attn_bias,
                dropout=config.attn_dropout,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                num_blocks=config.num_hidden_layers,
                round_proj_up_to_multiple_of=config.round_proj_up_to_multiple_of,
                round_proj_up_dim_up=config.round_proj_up_dim_up,
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
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        attnres_states: list[torch.Tensor] | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None, None]:
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
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        return hidden_states, attentions, past_key_values, None


class MLSTMPreTrainedModel(FLAPreTrainedModel):

    config_class = MLSTMConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['MLSTMBlock']
    _supports_cache_class = True

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, ShortConvolution):
            module.reset_parameters()
        elif isinstance(module, MLSTM):
            # small initialization from GPT-NeoX
            nn.init.kaiming_normal_(module.proj_up.weight, a=2)

            # depth-scaled output initialization from GPT-NeoX
            nn.init.kaiming_normal_(module.proj_down.weight, mode="fan_out", nonlinearity="linear")
            with torch.no_grad():
                module.proj_down.weight *= 2 / module.num_blocks
                for proj in (module.q_proj, module.k_proj, module.v_proj):
                    proj.weight *= (proj.weight.shape[-1] / module.hidden_size) ** 0.5
                bias = module.fgate.bias
                bias.copy_(torch.linspace(start=3.0, end=6.0, steps=bias.numel(), device=bias.device, dtype=bias.dtype))

            nn.init.ones_(module.learnable_skip)
            nn.init.zeros_(module.fgate.weight)
            nn.init.zeros_(module.igate.weight)
            nn.init.normal_(module.igate.bias, mean=0.0, std=0.1)
        else:
            super()._init_weights(
                module=module,
                prenorm_residual_strategy=prenorm_residual_strategy,
                num_residuals_per_layer=num_residuals_per_layer,
            )


class MLSTMModel(MLSTMPreTrainedModel, FLAModel):

    def __init__(self, config: MLSTMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MLSTMBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.use_attnres = False
        self.gradient_checkpointing = False

        self.post_init()


class MLSTMForCausalLM(MLSTMPreTrainedModel, FLAForCausalLM, FLAUnsupportedCacheGenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = MLSTMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()
