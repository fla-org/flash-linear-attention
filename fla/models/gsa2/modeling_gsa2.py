# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.gsa2 import GatedSlotAttention2
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.gsa2.configuration_gsa2 import GSA2Config
from fla.models.modeling_utils import FLAForCausalLM, FLAModel, FLAPreTrainedModel, get_attention
from fla.modules import GatedMLP as GSA2MLP
from fla.modules import RMSNorm
from fla.ops.attnres import fused_attnres

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


class GSA2Block(GradientCheckpointingLayer):

    def __init__(self, config: GSA2Config, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = get_attention(config=config, layer_idx=layer_idx)
        if self.attn is None:
            self.attn = GatedSlotAttention2(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_v_heads=config.num_v_heads,
                num_slots=config.num_slots,
                level2_axis=config.level2_axis,
                w_rank=config.w_rank,
                use_slot_proj=config.use_slot_proj,
                w_scale=config.w_scale,
                w_clip=config.w_clip,
                use_w_l2norm=config.use_w_l2norm,
                w_l2norm_eps=config.w_l2norm_eps,
                use_short_conv=config.use_short_conv,
                use_w_conv=config.use_w_conv,
                conv_size=config.conv_size,
                conv_bias=config.conv_bias,
                oja_out_act=config.oja_out_act,
                oja_scale=config.oja_scale,
                gdn_scale=config.gdn_scale,
                use_oja_q_l2norm=config.use_oja_q_l2norm,
                use_oja_k_l2norm=config.use_oja_k_l2norm,
                use_gdn_qk_l2norm_in_kernel=config.use_gdn_qk_l2norm_in_kernel,
                oja_lower_bound=config.oja_lower_bound,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
            )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = GSA2MLP(
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
                # with one residual, attention mixing reduces to the prenorm
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


class GSA2PreTrainedModel(FLAPreTrainedModel):

    config_class = GSA2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GSA2Block']
    _supports_cache_class = True

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, GatedSlotAttention2) and next(module.parameters()).device.type != 'meta':
            with torch.no_grad():
                for name in ('A1_log', 'A2_log'):
                    param = getattr(module, name)
                    if not getattr(param, '_is_hf_initialized', False):
                        param.copy_(nn.init.uniform_(param, a=1, b=16).log())
                    param._no_weight_decay = True
                if not getattr(module.dt1_bias, '_is_hf_initialized', False):
                    module.dt1_bias.zero_()
                module.dt1_bias._no_weight_decay = True
                if not getattr(module.dt2_bias, '_is_hf_initialized', False):
                    dt = torch.exp(
                        nn.init.uniform_(module.dt2_bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001),
                    ).clamp(min=1e-4)
                    # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                    inv_dt = dt + torch.log(-torch.expm1(-dt))
                    module.dt2_bias.copy_(inv_dt)
                module.dt2_bias._no_weight_decay = True

        super()._init_weights(
            module=module,
            prenorm_residual_strategy=prenorm_residual_strategy,
            num_residuals_per_layer=num_residuals_per_layer,
        )


class GSA2Model(GSA2PreTrainedModel, FLAModel):

    def __init__(self, config: GSA2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GSA2Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            self.res_proj = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
            self.res_norm = nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.norm_eps)
            self.res_proj._is_attnres_proj = True

        self.gradient_checkpointing = False

        self.post_init()


class GSA2ForCausalLM(GSA2PreTrainedModel, FLAForCausalLM, FLAUnsupportedCacheGenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = GSA2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor | None = 0,
        num_items_in_batch: int | torch.Tensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast:
        cu_seqlens = kwargs.get("cu_seqlens")
        if labels is not None and cu_seqlens is not None:
            ignore_index = self.criterion.ignore_index if self.criterion is not None else -100
            labels = labels.clone()
            starts = cu_seqlens[1:-1].to(device=labels.device, dtype=torch.long)
            # the shared loss shifts labels left, so exclude the first target of each packed sequence
            labels.reshape(-1).index_fill_(0, starts[starts < labels.numel()], ignore_index)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )
