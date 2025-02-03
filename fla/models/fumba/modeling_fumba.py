# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging, Cache

from fla.layers.dswattn import DualSlidingWindowAttention
from fla.models.mamba.modeling_mamba import MambaCache, MambaMixer
from fla.models.fumba.configuration_fumba import FumbaConfig
from fla.modules import (FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss,
                         RMSNorm)
from fla.modules.activations import swiglu_linear

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)

class FumbaCache(MambaCache):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the states of each layer as the tensor of shape `[batch_size, key_dim, value_dim]`.
    """

    def __init__(
        self,
        config: FumbaConfig,
        batch_size: int = None,
        dtype: torch.dtype = torch.float16,
        device: Optional[Union[torch.device, str]] = None,
        seen_tokens: int = 0
    ):
        super().__init__(config, batch_size, dtype, device)
        self.window_size = config.window_size
        self.att_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = [(None, None)] * config.num_hidden_layers
        self.ssm_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = [(None, None)] * config.num_hidden_layers

    def update_att_state(
        self, layer_idx: int, attn_keys: torch.Tensor, attn_values: torch.Tensor, ssm_keys: torch.Tensor, ssm_values: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        past_att_keys, past_att_values = self.att_key_values[layer_idx]
        past_ssm_keys, past_ssm_values = self.ssm_key_values[layer_idx]
        if past_att_keys is not None:
            input_len = attn_keys.shape[-2]
            past_len = past_att_keys.shape[-2]
            if self.window_size is not None and past_len + input_len > self.window_size:
                if past_len > self.window_size:
                    past_att_keys = past_att_keys[..., -self.window_size:]
                    past_att_values = past_att_values[..., -self.window_size:]
                    past_ssm_keys = past_ssm_keys[..., -self.window_size:]
                    past_ssm_values = past_ssm_values[..., -self.window_size:]
                past_att_keys = past_att_keys.roll(-input_len, dims=-2)
                past_att_values = past_att_values.roll(-input_len, dims=-2)
                past_ssm_keys = past_ssm_keys.roll(-input_len, dims=-2)
                past_ssm_values = past_ssm_values.roll(-input_len, dims=-2)
                past_att_keys[..., -input_len:] = attn_keys
                past_att_values[..., -input_len:] = attn_values
                past_ssm_keys[..., -input_len:] = ssm_keys
                past_ssm_values[..., -input_len:] = ssm_values
            else:
                past_att_keys = torch.cat([past_att_keys, attn_keys], dim=-2)
                past_att_values = torch.cat([past_att_values, attn_values], dim=-2)
                past_ssm_keys = torch.cat([past_ssm_keys, ssm_keys], dim=-2)
                past_ssm_values = torch.cat([past_ssm_values, ssm_values], dim=-2)
            self.att_key_values[layer_idx] = (past_att_keys, past_att_values)
            self.ssm_key_values[layer_idx] = (past_ssm_keys, past_ssm_values)
        else:
            self.att_key_values[layer_idx] = (attn_keys, attn_values)
            self.ssm_key_values[layer_idx] = (ssm_keys, ssm_values)

        return self.att_key_values[layer_idx], self.ssm_key_values[layer_idx]
    
    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()
        self.att_key_values = [(None, None)] * self.config.num_hidden_layers
        self.ssm_key_values = [(None, None)] * self.config.num_hidden_layers

class FumbaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        hidden_act: str = 'swish'
    ) -> FumbaMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        self.hidden_ratio = hidden_ratio

        self.intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
        self.intermediate_size = 256 * ((self.intermediate_size + 256 - 1) // 256)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Unpack[Dict],
    ) -> torch.Tensor:
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        return swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)


class FumbaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.pre_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.ssm = MambaMixer(config, layer_idx=layer_idx)
        self.ssm_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.att = DualSlidingWindowAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            window_size=config.window_size,
            max_position_embeddings=config.max_position_embeddings,
            norm_first=config.fuse_norm,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx
        )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.mlp = FumbaMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            hidden_act=config.hidden_act
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[FumbaCache] = None,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.pre_norm(hidden_states)
        ssm_states = self.ssm(hidden_states, cache_params=cache_params)
        ssm_states = self.ssm_norm(ssm_states)
        hidden_states, cache_params = self.att(hidden_states, ssm_states, cache_params=cache_params, **kwargs)
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        return hidden_states


class FumbaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FumbaConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["FumbaBlock"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                nn.init.constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                nn.init.uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = torch.exp(
                torch.rand(self.config.intermediate_size)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_proj.bias.copy_(inv_dt)
            module.dt_proj.bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_layers)


@dataclass
class FumbaOutput(ModelOutput):
    """
    Class for the Fumba model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`FumbaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*,
            returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[FumbaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class FumbaCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`FumbaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*,
            returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[FumbaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class FumbaModel(FumbaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([FumbaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = RMSNorm(config.hidden_size, eps=config.norm_eps)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[FumbaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, FumbaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = FumbaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__,
                    hidden_states,
                    cache_params,
                    **kwargs
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    **kwargs
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return FumbaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class FumbaForCausalLM(FumbaPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = FumbaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cache_params:
        Optional[FumbaCache] = None,
        inputs_embeds=None,
        attention_mask=None,
        use_cache: Optional[bool] = True,
        num_logits_to_keep: Optional[int] = None,
        **kwargs: Unpack[Dict]
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if num_logits_to_keep is not None:
            model_inputs['num_logits_to_keep'] = num_logits_to_keep

        model_inputs.update({
            'cache_params': cache_params,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'num_logits_to_keep': num_logits_to_keep,
        })
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[FumbaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        num_logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, FumbaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        fumba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = fumba_outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training
        logits = None if fuse_linear_and_cross_entropy else self.lm_head(hidden_states[:, -num_logits_to_keep:])

        loss = None
        if labels is not None:
            if self.config.fuse_cross_entropy:
                if fuse_linear_and_cross_entropy:
                    loss_fct = FusedLinearCrossEntropyLoss()
                else:
                    loss_fct = FusedCrossEntropyLoss(inplace_backward=True)
            else:
                loss_fct = nn.CrossEntropyLoss()
            # Enable model parallelism
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = loss_fct(hidden_states.view(-1, self.config.hidden_size),
                                labels.view(-1),
                                self.lm_head.weight,
                                self.lm_head.bias)
            else:
                loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + fumba_outputs[1:]
            return (loss,) + output if loss is not None else output

        return FumbaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=fumba_outputs.cache_params,
            hidden_states=fumba_outputs.hidden_states,
        )
