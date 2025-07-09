# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from fla.layers.attn import Attention
from fla.layers import MomGatedDeltaNet
from fla.models.mom_gated_deltanet.configuration_mom_gated_deltanet import \
    MomGatedDeltaNetConfig
from fla.models.utils import Cache
from fla.modules import (FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss,
                         RMSNorm)
from fla.modules.activations import swiglu_linear
from fla.modules.layernorm import rms_norm_linear

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


logger = logging.get_logger(__name__)


class MomGatedDeltaNetMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        norm_first: bool = True,
        norm_eps: float = 1e-5
    ) -> MomGatedDeltaNetMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.norm_first = norm_first

        if norm_first:
            self.norm = RMSNorm(hidden_size=hidden_size, eps=norm_eps)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Unpack[Dict],
    ) -> torch.Tensor:
        if self.norm_first:
            x = rms_norm_linear(x, self.norm.weight, self.norm.bias, self.gate_proj.weight, self.gate_proj.bias)
        else:
            x = self.gate_proj(x)
        gate, y = x.chunk(2, -1)
        return swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)


class MomGatedDeltaNetBlock(nn.Module):
    def __init__(self, config: MomGatedDeltaNetConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if not config.norm_first:
            self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                window_size=config.attn['window_size'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx
            )
        else:
            self.attn = MomGatedDeltaNet(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                use_gate=config.use_gate,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                norm_first=config.norm_first,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                num_memories=config.num_memories,
                topk=config.topk,
                capacity=config.capacity,
                shared_mem=config.shared_mem,
                single_kv_proj=config.single_kv_proj
            )
        if not config.norm_first:
            self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.mlp = MomGatedDeltaNetMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            norm_first=config.norm_first,
            norm_eps=config.norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        if hasattr(self, 'attn_norm'):
            hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values, router_logits = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if hasattr(self, 'mlp_norm'):
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values, router_logits)

        return outputs


class MomGatedDeltaNetPreTrainedModel(PreTrainedModel):

    config_class = MomGatedDeltaNetConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['GatedDeltaNetBlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)

@dataclass
class MomGatedDeltaNetOutputWithPast(BaseModelOutputWithPast):
    router_logits: Optional[Tuple[torch.FloatTensor, ...]] = None

class MomGatedDeltaNetModel(MomGatedDeltaNetPreTrainedModel):

    def __init__(self, config: MomGatedDeltaNetConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MomGatedDeltaNetBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

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
            warnings.warn("`GatedDeltaNetModel` does not `output_attentions` now, setting it to `False`.")
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
        all_router_logits = ()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values, router_logits = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    **kwargs
                )
            else:
                hidden_states, attentions, past_key_values, router_logits = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs
                )

            if output_attentions:
                all_attns += (attentions,)
            all_router_logits += (router_logits,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return MomGatedDeltaNetOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            router_logits=all_router_logits
        )

@dataclass
class MomGatedDeltaNetCausalLMOutputWithPast(CausalLMOutputWithPast):
    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[Tuple[torch.FloatTensor, ...]] = None

class MomGatedDeltaNetForCausalLM(MomGatedDeltaNetPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MomGatedDeltaNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_memories = config.num_memories
        self.topk = config.topk
        self.aux_loss_scale = config.aux_loss_scale

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

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        num_logits_to_keep: Optional[int] = 0,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is passed along.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        if num_logits_to_keep is not None:
            model_inputs['num_logits_to_keep'] = num_logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'num_logits_to_keep': num_logits_to_keep,
        })
        return model_inputs

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
        num_logits_to_keep: Optional[int] = 0,
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
        logits = None if fuse_linear_and_cross_entropy else self.lm_head(hidden_states[:, -num_logits_to_keep:])

        loss = None
        aux_loss = None
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

            valid_router_logits = tuple(
                logits
                for logits in (outputs.router_logits if return_dict else outputs[-1])
                if logits is not None
            )
            aux_loss = load_balancing_loss_func(
                valid_router_logits,
                self.num_memories,
                self.topk,
                use_layer_wise_balance=self.config.use_layer_wise_balance,  # ✨
            )
            aux_loss *= self.aux_loss_scale

            loss += aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MomGatedDeltaNetCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            aux_loss=aux_loss
        )


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple],
    num_memories: torch.Tensor = None,
    top_k=2,
    use_layer_wise_balance=False,
) -> torch.FloatTensor:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of tensors. Shape: [batch_size, seqeunce_length, num_memories].
        num_memories (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or (
        isinstance(gate_logits, Iterable) and len(gate_logits) == 0
    ):
        return 0

    # ✨ Here is the fix for balance loss in Mixtral.
    # We should calculate the balance loss in a layer-wise manner otherwise it may lead to degenerated solutions.
    if use_layer_wise_balance:
        if not isinstance(gate_logits, Iterable):
            gate_logits = (gate_logits,)
    else:
        if isinstance(gate_logits, Iterable):
            gate_logits = (torch.cat(gate_logits, dim=0),)
        else:
            gate_logits = (gate_logits,)

    all_balance_losses = []

    for logits in gate_logits:
        routing_weights, selected_experts = torch.topk(logits, top_k, dim=-1)
        routing_weights = routing_weights.softmax(dim=-1)
        routing_weights_full = torch.zeros_like(logits).scatter(-1, selected_experts, routing_weights)


        # cast the expert indices to int64, otherwise one-hot encoding will fail
        if selected_experts.dtype != torch.int64:
            selected_experts = selected_experts.to(torch.int64)

        if len(selected_experts.shape) == 2:
            selected_experts = selected_experts.unsqueeze(2)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_memories)

        # For a given token, determine if it was routed to a given expert.
        expert_mask = torch.max(expert_mask, axis=-2).values

        # cast to float32 otherwise mean will fail
        expert_mask = expert_mask.to(torch.float32)
        tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

        router_prob_per_group_and_expert = torch.mean(routing_weights_full, axis=-2)

        # ✨ balance loss for this layer
        balance_loss = torch.mean(
            tokens_per_group_and_expert * router_prob_per_group_and_expert
        ) * (num_memories**2)
        all_balance_losses.append(balance_loss.reshape(1))

    all_balance_losses = torch.cat(all_balance_losses).mean()  # ✨

    return all_balance_losses
