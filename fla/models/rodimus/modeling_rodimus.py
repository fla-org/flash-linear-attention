# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.layers.rodimus import RodimusAttention, SlidingWindowSharedKeyAttention, align_multiple
from fla.models.rodimus.configuration_rodimus import RodimusConfig
from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as RodimusMLP
from fla.modules import RMSNorm
from fla.modules.l2warp import l2_warp

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)


class RodimusBlock(nn.Module):
    def __init__(self, config: RodimusConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.block_type = config.block_type
        self.block_residual_in_fp32 = config.residual_in_fp32
        self.residual_in_fp32 = config.residual_in_fp32
        self.fuse_norm = config.fuse_norm

        self._is_ori_attn = False

        if config.intermediate_size is None:
            intermediate_size = align_multiple(int(config.hidden_ratio * config.hidden_size), 8)
        else:
            intermediate_size = config.intermediate_size

        mlp_cls = partial(
            RodimusMLP,
            hidden_size=config.hidden_size,
            hidden_ratio=None,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu
        )
        norm_cls = partial(
            RMSNorm if self.fuse_norm else nn.RMSNorm,
            config.hidden_size,
            eps=config.norm_eps,
        )

        if config.attn is not None and layer_idx in config.attn['layers']:
            self._is_ori_attn = True
            self.attn_norm = norm_cls()
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                qkv_bias=config.attn['qkv_bias'],
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx
            )

            self.mlp_norm = norm_cls()
            self.mlp = mlp_cls()
        else:
            self.mixer_norm = norm_cls()
            self.mixer = RodimusAttention(
                block_type=config.block_type,
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                input_gate_low_rank=config.input_gate_low_rank,
                expand_ratio=config.expand_ratio,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                k_norm_eps=config.k_norm_eps,
                residual_in_fp32=config.residual_in_fp32,
                layer_idx=layer_idx
            )

            if self.block_type == "rodimus_plus":
                self.ska_attn_norm = norm_cls()
                self.ska_attn = SlidingWindowSharedKeyAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.ska_attn['num_heads'],
                    qkv_bias=config.ska_attn['qkv_bias'],
                    qk_norm=config.ska_attn['qk_norm'],
                    window_size=config.ska_attn['window_size'],
                    rope_theta=config.ska_attn['rope_theta'],
                    max_position_embeddings=config.max_position_embeddings,
                    layer_idx=layer_idx
                )

                self.mlp_norm = norm_cls()
                self.mlp = mlp_cls()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        residual: Optional[torch.Tensor] = None,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if self.block_residual_in_fp32 and self.layer_idx > 0:
            assert residual is not None, 'Residual must be passed in when setting `block_residual_in_fp32=True`'

        if self._is_ori_attn:
            if self.block_residual_in_fp32:
                hidden_states, residual = self.attn_norm(
                    hidden_states,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32
                )
            else:
                residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
                hidden_states = self.attn_norm(hidden_states)

            hidden_states, attentions, past_key_values = self.attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs
            )
            if self.fuse_norm:
                hidden_states, residual = self.mlp_norm(
                    hidden_states,
                    residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32
                )
            else:
                hidden_states = residual + hidden_states
                residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
                hidden_states = self.mlp_norm(hidden_states.to(self.mlp_norm.weight.dtype))

            hidden_states = self.mlp(hidden_states, **kwargs)
        else:
            if self.block_residual_in_fp32:
                hidden_states, residual = self.mixer_norm(
                    hidden_states,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32
                )
            else:
                residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
                hidden_states = self.mixer_norm(hidden_states)

            hidden_states, attentions, past_key_values = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs
            )

            if self.block_type == "rodimus_plus":
                past_key_values, rodimus_caches = past_key_values

                if self.fuse_norm:
                    hidden_states, residual = self.ska_attn_norm(
                        hidden_states,
                        residual,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32
                    )
                else:
                    hidden_states = residual + hidden_states
                    residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
                    hidden_states = self.ska_attn_norm(hidden_states.to(dtype=self.ska_attn_norm.weight.dtype))

                hidden_states, attentions, past_key_values = self.ska_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    rodimus_caches=rodimus_caches,
                    **kwargs
                )

                if self.fuse_norm:
                    hidden_states = self.mlp_norm(
                        hidden_states,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32
                    )
                else:
                    hidden_states = residual + hidden_states
                    hidden_states = self.mlp_norm(hidden_states.to(dtype=self.mlp_norm.weight.dtype))

                hidden_states = self.mlp(hidden_states, **kwargs)

        if self.block_residual_in_fp32:
            hidden_states = (hidden_states, residual)
        else:
            hidden_states = (residual + hidden_states).to(dtype=hidden_states.dtype)

        outputs = (hidden_states, attentions, past_key_values)
        return outputs


class RodimusPreTrainedModel(PreTrainedModel):

    config_class = RodimusConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['RodimusBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        if self.config.block_type == "rodimus":
            self.num_residuals_per_layer = 1
        elif self.config.block_type == "rodimus_plus":
            self.num_residuals_per_layer = 3
        else:
            raise NotImplementedError()

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: Optional[str] = None,
    ):
        num_residuals_per_layer = self.num_residuals_per_layer

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        sigmoid_bias_max = 0.999
        sigmoid_bias_min = 0.9
        max_ = 1 - sigmoid_bias_min
        min_ = 1 - sigmoid_bias_max
        g_gate_bias = torch.exp(
            torch.rand(self.config.expand_ratio) * (math.log(max_) - math.log(min_))
            + math.log(min_)
        ).clamp(min=1e-4)
        g_gate_bias = g_gate_bias + torch.log(-torch.expm1(-g_gate_bias))
        tau_gate_bias = torch.logit(torch.empty((self.config.expand_ratio, )).uniform_(1/16, 0.9))

        if hasattr(module, 'i_gate_proj'):
            nn.init.xavier_uniform_(module.i_gate_proj[0].weight, gain=2 ** -2.5)
            nn.init.xavier_uniform_(module.i_gate_proj[1].weight, gain=2 ** -2.5)
            nn.init.zeros_(module.i_gate_proj[1].bias)
        if hasattr(module, 'g_gate_proj'):
            nn.init.xavier_uniform_(module.g_gate_proj.weight, gain=2 ** -2.5)
            with torch.no_grad():
                if not isinstance(module.g_gate_proj.bias, DTensor):
                    module.g_gate_proj.bias.copy_(g_gate_bias)
                else:
                    logger.warning_once("`g_gate_proj.bias` is a DTensor, skipping initialization")

        if hasattr(module, 'tau_gate_proj'):
            nn.init.xavier_uniform_(module.tau_gate_proj.weight, gain=2 ** -2.5)
            with torch.no_grad():
                if not isinstance(module.tau_gate_proj.bias, DTensor):
                    module.tau_gate_proj.bias.copy_(tau_gate_bias)
                else:
                    logger.warning_once("`tau_gate_proj.bias` is a DTensor, skipping initialization")

        if prenorm_residual_strategy is not None:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")


class RodimusModel(RodimusPreTrainedModel):

    def __init__(self, config: RodimusConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.block_residual_in_fp32 = config.block_residual_in_fp32

        if config.block_residual_in_fp32:
            if not config.residual_in_fp32:
                warning_message = (
                    "`residual_in_fp32=False` is incompatible with `block_residual_in_fp32=True`. "
                    "Setting `residual_in_fp32=True`..."
                )
                logger.warning_once(warning_message)
                config.residual_in_fp32 = True
            if not config.fuse_norm:
                logger.warning_once(
                    '`fuse_norm=False` is incompatible with `block_residual_in_fp32=True` Setting `fuse_norm=True`...')
                config.fuse_norm = True

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([RodimusBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if output_attentions:
            warnings.warn("`RodimusModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        residual = None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    residual,
                    **kwargs
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    residual=residual,
                    **kwargs
                )

            if self.block_residual_in_fp32:
                hidden_states, residual = hidden_states
            else:
                residual = None

            if output_attentions:
                all_attns += (attentions,)

        if self.block_residual_in_fp32:
            hidden_states = self.norm(
                hidden_states,
                residual=residual,
                prenorm=False,
                residual_in_fp32=True,
            )
        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class RodimusForCausalLM(RodimusPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RodimusModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
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
            else:
                raise exception

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is not empty.
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(past_key_values) == 0:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
        })
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training

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
