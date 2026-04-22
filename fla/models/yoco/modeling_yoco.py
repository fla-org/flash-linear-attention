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
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.yoco_attn import YOCOCrossAttention, YOCOGatedRetention, YOCOSharedKVBuilder
from fla.models.utils import Cache, FLAGenerationMixin
from fla.models.yoco.configuration_yoco import YOCOConfig
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as YOCOMLP
from fla.modules.l2warp import l2_warp

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


logger = logging.get_logger(__name__)


def _checkpointed_yoco_self_layer_forward(
    layer: YOCOSelfDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | list[torch.FloatTensor] | None,
    use_cache: bool,
    output_attentions: bool,
    model_kwargs: dict,
):
    return layer(
        hidden_states,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        **model_kwargs,
    )


def _checkpointed_yoco_cross_layer_forward(
    layer: YOCOCrossDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    shared_k: torch.Tensor,
    shared_v: torch.Tensor,
    output_attentions: bool,
    model_kwargs: dict,
):
    return layer(
        hidden_states,
        shared_k=shared_k,
        shared_v=shared_v,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        **model_kwargs,
    )


def _select_last_query_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if hidden_states.shape[1] == 1 or attention_mask is None:
        return hidden_states[:, -1:]

    query_attention_mask = attention_mask[:, -hidden_states.shape[1]:].to(dtype=torch.bool)
    last_token_offsets = query_attention_mask.shape[1] - 1 - query_attention_mask.flip(-1).to(torch.int64).argmax(-1)
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, last_token_offsets].unsqueeze(1)


class YOCOSelfDecoderLayer(nn.Module):
    def __init__(self, config: YOCOConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self_decoder_attn = config.self_decoder_attn

        if self_decoder_attn['type'] == 'gated_deltanet':
            self.attn = GatedDeltaNet(
                mode=self_decoder_attn['mode'],
                hidden_size=config.hidden_size,
                expand_v=self_decoder_attn['expand_v'],
                head_dim=self_decoder_attn['head_dim'],
                num_heads=self_decoder_attn['num_heads'],
                num_v_heads=self_decoder_attn['num_v_heads'],
                use_gate=self_decoder_attn['use_gate'],
                use_short_conv=self_decoder_attn['use_short_conv'],
                allow_neg_eigval=self_decoder_attn['allow_neg_eigval'],
                conv_size=self_decoder_attn['conv_size'],
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
            )
        elif self_decoder_attn['type'] == 'gated_retention':
            self.attn = YOCOGatedRetention(
                mode=self_decoder_attn['mode'],
                hidden_size=config.hidden_size,
                num_heads=self_decoder_attn['num_heads'],
                rope_theta=self_decoder_attn['rope_theta'],
                rope_inv_freq=self_decoder_attn['rope_inv_freq'],
                max_position_embeddings=config.max_position_embeddings,
                gate_logit_normalizer=self_decoder_attn['gate_logit_normalizer'],
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
            )
        elif self_decoder_attn['type'] == 'swa':
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=self_decoder_attn['num_heads'],
                num_kv_heads=self_decoder_attn['num_v_heads'],
                qkv_bias=self_decoder_attn['qkv_bias'],
                window_size=self_decoder_attn['window_size'],
                rope_theta=self_decoder_attn['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
            )
        else:
            raise ValueError(f"Unsupported self_decoder_attn['type']: {self_decoder_attn['type']}")

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = YOCOMLP(
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
        **kwargs: Unpack[dict]
    ) -> tuple[
        torch.FloatTensor,
        tuple[torch.FloatTensor, torch.FloatTensor] | None,
        Cache | list[torch.FloatTensor] | None,
    ]:
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
        return hidden_states, attentions, past_key_values


class YOCOCrossDecoderLayer(nn.Module):
    def __init__(self, config: YOCOConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        cross_decoder_attn = config.cross_decoder_attn
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = YOCOCrossAttention(
            hidden_size=config.hidden_size,
            num_heads=cross_decoder_attn['num_heads'],
            num_kv_heads=cross_decoder_attn['num_kv_heads'],
            qkv_bias=cross_decoder_attn['qkv_bias'],
            qk_norm=cross_decoder_attn['qk_norm'],
            window_size=cross_decoder_attn['window_size'],
            rope_theta=cross_decoder_attn['rope_theta'],
            rope_inv_freq=cross_decoder_attn['rope_inv_freq'],
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = YOCOMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_k: torch.Tensor,
        shared_v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict]
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions = self.attn(
            hidden_states=hidden_states,
            shared_k=shared_k,
            shared_v=shared_v,
            attention_mask=attention_mask,
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
        return hidden_states, attentions


class YOCOPreTrainedModel(PreTrainedModel):

    config_class = YOCOConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['YOCOSelfDecoderLayer', 'YOCOCrossDecoderLayer']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @staticmethod
    def _init_yoco_linear(module: nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    @staticmethod
    def _init_yoco_xavier(module: nn.Linear, gain: float):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, GatedDeltaNet):
            A = torch.empty(module.num_v_heads, dtype=torch.float32).uniform_(0, 16)
            with torch.no_grad():
                if not isinstance(module.A_log, DTensor):
                    module.A_log.copy_(torch.log(A))
                else:
                    logger.warning_once("`A_log` is a DTensor, skipping initialization")
            module.A_log._no_weight_decay = True

            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(module.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                if not isinstance(module.dt_bias, DTensor):
                    module.dt_bias.copy_(inv_dt)
                else:
                    logger.warning_once("`dt_bias` is a DTensor, skipping initialization")
            module.dt_bias._no_weight_decay = True
            module.dt_bias._no_reinit = True

        elif isinstance(module, YOCOMLP):
            self._init_yoco_linear(module.gate_proj)
            self._init_yoco_linear(module.up_proj)
            self._init_yoco_linear(module.down_proj)

        elif isinstance(module, YOCOGatedRetention):
            for proj in (module.q_proj, module.k_proj, module.v_proj, module.g_proj, module.gk_proj):
                self._init_yoco_xavier(proj, gain=2 ** -2.5)
            self._init_yoco_xavier(module.o_proj, gain=2 ** -1)

        elif isinstance(module, YOCOSharedKVBuilder):
            self._init_yoco_linear(module.k_proj)
            self._init_yoco_linear(module.v_proj)

        elif isinstance(module, YOCOCrossAttention):
            self._init_yoco_linear(module.q_proj)
            self._init_yoco_linear(module.o_proj)

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim ** -0.5)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")


class YOCOModel(YOCOPreTrainedModel):

    def __init__(self, config: YOCOConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_scale = math.sqrt(config.hidden_size)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.self_layers = nn.ModuleList(
            [YOCOSelfDecoderLayer(config, layer_idx) for layer_idx in range(config.num_self_decoder_layers)]
        )
        self.shared_kv_builder = None
        self.cross_layers = nn.ModuleList([])
        if config.num_hidden_layers > config.num_self_decoder_layers:
            cross_decoder_attn = config.cross_decoder_attn
            self.shared_kv_builder = YOCOSharedKVBuilder(
                hidden_size=config.hidden_size,
                num_heads=cross_decoder_attn['num_heads'],
                num_kv_heads=cross_decoder_attn['num_kv_heads'],
                qkv_bias=cross_decoder_attn['qkv_bias'],
                rope_theta=cross_decoder_attn['rope_theta'],
                rope_inv_freq=cross_decoder_attn['rope_inv_freq'],
                max_position_embeddings=config.max_position_embeddings,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=config.num_self_decoder_layers,
            )
            self.cross_layers = nn.ModuleList([
                YOCOCrossDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_self_decoder_layers, config.num_hidden_layers)
            ])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    @property
    def layers(self):
        return list(self.self_layers) + list(self.cross_layers)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

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
        skip_cross_decoder: bool = False,
        **kwargs: Unpack[dict]
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            logger.warning_once("`YOCOModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if skip_cross_decoder and self.cross_layers and output_hidden_states:
            raise ValueError(
                "`skip_cross_decoder=True` is incompatible with `output_hidden_states=True` because "
                "the skip path only computes cross-decoder outputs for the final query token."
            )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            # Match official YOCO, which scales token embeddings before entering the decoder stack.
            inputs_embeds = self.embed_scale * self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        for layer in self.self_layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            use_checkpoint = self.gradient_checkpointing and self.training
            if use_checkpoint:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    _checkpointed_yoco_self_layer_forward,
                    layer,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    kwargs,
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs,
                )

            if output_attentions:
                all_attns += (attentions,)

        shared_k = None
        shared_v = None
        if self.shared_kv_builder is not None:
            shared_k, shared_v, past_key_values = self.shared_kv_builder(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        if skip_cross_decoder and self.cross_layers:
            # During prompt prefilling, only the final query token is needed to produce the next-token logits.
            hidden_states = _select_last_query_hidden_states(hidden_states, attention_mask)

        for layer in self.cross_layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            use_checkpoint = self.gradient_checkpointing and self.training
            if use_checkpoint:
                hidden_states, attentions = self._gradient_checkpointing_func(
                    _checkpointed_yoco_cross_layer_forward,
                    layer,
                    hidden_states,
                    attention_mask,
                    shared_k,
                    shared_v,
                    output_attentions,
                    kwargs,
                )
            else:
                hidden_states, attentions = layer(
                    hidden_states,
                    shared_k=shared_k,
                    shared_v=shared_v,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    **kwargs,
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

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


class YOCOForCausalLM(YOCOPreTrainedModel, FLAGenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = YOCOModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            raise exception

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
        logits_to_keep: int | None = None,
        cache_position: torch.LongTensor | None = None,
        skip_cross_decoder: bool | None = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            cache_position=cache_position,
            **kwargs,
        )

        if skip_cross_decoder is None:
            has_cache = past_key_values is not None and hasattr(past_key_values, '__len__') and len(past_key_values) > 0
            model_input_ids = model_inputs.get('input_ids')
            model_inputs_embeds = model_inputs.get('inputs_embeds')
            input_length = (
                model_input_ids.shape[1]
                if model_input_ids is not None
                else (model_inputs_embeds.shape[1] if model_inputs_embeds is not None else 0)
            )
            skip_cross_decoder = bool(self.model.cross_layers) and not has_cache and input_length > 1

        model_inputs['skip_cross_decoder'] = skip_cross_decoder
        return model_inputs

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
        logits_to_keep: int | None = 0,
        skip_cross_decoder: bool = False,
        **kwargs: Unpack[dict]
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if skip_cross_decoder and self.model.cross_layers:
            incompatible_args = []
            if labels is not None:
                incompatible_args.append("`labels`")
            if output_hidden_states:
                incompatible_args.append("`output_hidden_states=True`")
            if incompatible_args:
                incompatible_args_str = " and ".join(incompatible_args)
                raise ValueError(
                    f"`skip_cross_decoder=True` is incompatible with {incompatible_args_str} because "
                    "the skip path only computes cross-decoder outputs for the final query token."
                )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            skip_cross_decoder=skip_cross_decoder,
            **kwargs,
        )

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training and labels is not None

        loss, logits = None, None
        if not fuse_linear_and_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

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
