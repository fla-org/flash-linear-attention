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
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.iso_kdn import IsotropicKalmanDeltaNetwork
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.iso_kdn.configuration_iso_kdn import IsoKDNConfig
from fla.models.modeling_utils import FLAForCausalLM, FLAModel, FLAPreTrainedModel, get_attention
from fla.modules import GatedMLP as IsoKDNMLP
from fla.modules import RMSNorm
from fla.ops.attnres import fused_attnres

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


logger = logging.get_logger(__name__)


class IsoKDNBlock(GradientCheckpointingLayer):
    def __init__(self, config: IsoKDNConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = get_attention(config=config, layer_idx=layer_idx)
        if self.attn is None:
            self.attn = IsotropicKalmanDeltaNetwork(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_v_heads=config.num_v_heads,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                omega_min=config.omega_min,
                r_min=config.r_min,
                info_scale=config.info_scale,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
            )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = IsoKDNMLP(
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


class IsoKDNPreTrainedModel(FLAPreTrainedModel):
    config_class = IsoKDNConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["IsoKDNBlock"]
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, IsotropicKalmanDeltaNetwork) and next(module.parameters()).device.type != "meta":
            with torch.no_grad():
                if not getattr(module.A_log, '_is_hf_initialized', False):
                    module.A_log.copy_(nn.init.uniform_(module.A_log, a=1, b=16).log())
                if not getattr(module.dt_bias, '_is_hf_initialized', False):
                    dt = torch.exp(
                        nn.init.uniform_(module.dt_bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001),
                    ).clamp(min=1e-4)
                    inv_dt = dt + torch.log(-torch.expm1(-dt))
                    module.dt_bias.copy_(inv_dt)
                module.dt_bias._is_hf_initialized = True
                for bias in (module.omega_bias, module.r_bias):
                    if not getattr(bias, '_is_hf_initialized', False):
                        bias.zero_()
                    bias._is_hf_initialized = True
                if not getattr(module.initial_precision_param, '_is_hf_initialized', False):
                    module.initial_precision_param.fill_(math.log(math.expm1(0.9)))
                module.initial_precision_param._is_hf_initialized = True
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if getattr(module, '_is_attnres_proj', False):
                # attnres pseudo-query (per-layer projection): zero init keeps
                # the initial softmax uniform (paper §5)
                nn.init.zeros_(module.weight)
            else:
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None and not getattr(module.bias, "_is_hf_initialized", False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py
            p = None
            if hasattr(module, "o_proj"):
                p = module.o_proj.weight
            elif hasattr(module, "down_proj"):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                if prenorm_residual_strategy == "rescale":
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == "zero":
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")


class IsoKDNModel(IsoKDNPreTrainedModel, FLAModel):
    def __init__(self, config: IsoKDNConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([IsoKDNBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            # top-level attnres aggregation params; `self.norm` still applies afterward
            self.res_proj = nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
            self.res_norm = nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.norm_eps)
            self.res_proj._is_attnres_proj = True

        self.gradient_checkpointing = False

        self.post_init()


class IsoKDNForCausalLM(IsoKDNPreTrainedModel, FLAForCausalLM, FLAUnsupportedCacheGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = IsoKDNModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        loss, logits = self._compute_logits_and_loss(
            hidden_states=hidden_states,
            labels=labels,
            logits_to_keep=logits_to_keep,
            num_items_in_batch=num_items_in_batch,
            fuse_linear_cross_entropy=self.config.fuse_cross_entropy and self.training,
        )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
