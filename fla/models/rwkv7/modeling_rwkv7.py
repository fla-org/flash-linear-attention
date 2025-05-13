# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.layers.rwkv7 import RWKV7Attention
from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, LayerNorm
from fla.modules.activations import ACT2FN
from fla.modules.token_shift import token_shift

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)


class RWKV7FeedForward(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'sqrelu',
        layer_idx: int = None,
        num_hidden_layers: int = None,
    ) -> RWKV7FeedForward:
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio)
            intermediate_size = 32 * ((intermediate_size + 32 - 1) // 32)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_k = nn.Parameter(torch.zeros(hidden_size))

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        for name, module in self.named_modules():
            module._in_rwkv_module = True

    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, RWKV7FeedForward):
            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (module.layer_idx / module.num_hidden_layers)  # 1 to ~0
                ddd = torch.ones(1, 1, module.hidden_size)
                for i in range(module.hidden_size):
                    ddd[0, 0, i] = i / module.hidden_size
                module.x_k.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0**4).squeeze()

            # Initialize key and value weights as in CMix_x070
            module.key.weight.data.uniform_(-0.5/(module.hidden_size**0.5), 0.5/(module.hidden_size**0.5))
            module.value.weight.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[Cache] = None,
        **kwargs
    ) -> torch.Tensor:
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        if x.shape[1] == 1 and state is not None and state[self.layer_idx]['ffn_state'] is not None:
            shifted = state[self.layer_idx]['ffn_state'].unsqueeze(1)
            delta = shifted - x
        elif state is not None and state[self.layer_idx]['ffn_state'] is not None:
            shifted = self.time_shift(x)
            shifted[:, 0] = state[self.layer_idx]['ffn_state'][-1]
            delta = shifted - x
        else:
            cu_seqlens = kwargs.get('cu_seqlens', None)
            delta = token_shift(x, cu_seqlens)
        if state is not None:
            # no need to update the offset twice
            state.update(ffn_state=x[:, -1], layer_idx=self.layer_idx, offset=0)
        return self.value(self.act_fn(self.key(x.addcmul(delta, self.x_k)))), state


class RWKV7Block(nn.Module):

    def __init__(
        self,
        config: RWKV7Config,
        layer_idx: int
    ) -> RWKV7Block:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        if config.norm_first and layer_idx == 0:
            self.pre_norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
                config.hidden_size,
                bias=config.norm_bias,
                eps=config.norm_eps
            )
        self.attn_norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps
        )
        if config.attn is not None and layer_idx in config.attn['layers']:
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
        else:
            self.attn = RWKV7Attention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                decay_low_rank_dim=config.decay_low_rank_dim,
                gate_low_rank_dim=config.gate_low_rank_dim,
                a_low_rank_dim=config.a_low_rank_dim,
                v_low_rank_dim=config.v_low_rank_dim,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
                value_dim=config.value_dim[layer_idx],
                num_hidden_layers=config.num_hidden_layers
            )
        self.ffn_norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps
        )
        self.ffn = RWKV7FeedForward(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_idx=layer_idx,
            num_hidden_layers=config.num_hidden_layers
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        v_first: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states
        hidden_states = self.attn_norm(residual)
        hidden_states, attentions, past_key_values, v_first = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            v_first=v_first,
            **kwargs
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.ffn_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.ffn_norm(hidden_states)
        hidden_states, past_key_values = self.ffn(hidden_states, attention_mask, past_key_values, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values, v_first)

        return outputs


class RWKV7PreTrainedModel(PreTrainedModel):

    config_class = RWKV7Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['RWKV7Block']
    _supports_cache_class = True
    _skip_keys_device_placement = ["past_key_values"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @torch.no_grad()
    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, nn.Embedding):
            # https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/src/model.py#L396C12-L399C58
            scale = -1e-4
            nn.init.uniform_(module.weight, a=scale, b=-scale)
        elif isinstance(module, nn.Linear) and hasattr(self, 'lm_head') and module is self.lm_head:
            # https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/src/model.py#L403
            if self.config.vocab_size > self.config.hidden_size:
                scale = 0.5 * math.sqrt(self.config.vocab_size / self.config.hidden_size)
            else:
                scale = 0.5
            original_dtype = module.weight.dtype
            module.weight.data = nn.init.orthogonal_(module.weight.data.to(torch.float32), gain=scale).to(original_dtype)
        # Init Attention parameters
        elif isinstance(module, (nn.Linear, nn.Conv1d)) and getattr(module, '_in_rwkv_module', False) is False:
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters') and getattr(module, '_in_rwkv_module', False) is False:
            module.reset_parameters()

        if rescale_prenorm_residual:
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
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class RWKV7Model(RWKV7PreTrainedModel):

    def __init__(self, config: RWKV7Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([RWKV7Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps
        )

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Override the load_state_dict method to handle migration from version 1 to version 2.
        Handles hierarchical keys like 'model.layers.0.attn.x_x'.
        """
        # Collect all layer indices from the state_dict keys
        layer_indices = set()
        for key in state_dict.keys():
            if key.startswith("model.layers."):
                # Extract the layer index from the key
                try:
                    layer_idx = int(key.split(".")[2])  # Extract the number after 'model.layers.'
                    layer_indices.add(layer_idx)
                except ValueError:
                    # Skip keys that don't match the expected format
                    continue

        # Sort the layer indices to process them in order
        sorted_layer_indices = sorted(layer_indices)

        # Migration logic for each layer
        for layer_idx in sorted_layer_indices:
            layer_prefix = f"model.layers.{layer_idx}"
            attn_prefix = f"{layer_prefix}.attn"

            # Check if the layer contains the old 'x_x' parameter
            if f"{attn_prefix}.x_x" in state_dict:
                logger.info(f"Migrating weights for layer {layer_idx} from RWKV7Attention version 1 to version 2...")
                # Extract the x_x parameter
                x_x = state_dict[f"{attn_prefix}.x_x"]
                with torch.no_grad():
                    # Create new parameters for version 2
                    state_dict[f"{attn_prefix}.x_r"] = x_x[0].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_w"] = x_x[1].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_k"] = x_x[2].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_v"] = x_x[3].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_a"] = x_x[4].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_g"] = x_x[5].unsqueeze(0).unsqueeze(0)

        # Call the parent method to load the modified state_dict
        try:
            super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            # If the parent method does not support `assign`, fall back to strict loading
            logger.warning(
                "`assign` parameter is not supported by the parent `load_state_dict` method. "
                "Falling back to default behavior."
            )
            super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`RWKV7Model` does not `output_attentions` now, setting it to `False`.")
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

        v_first = torch.zeros_like(hidden_states)
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values, v_first = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    v_first,
                    **kwargs
                )
            else:
                hidden_states, attentions, past_key_values, v_first = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    v_first=v_first,
                    **kwargs
                )

            if output_attentions:
                all_attns += (attentions,)

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


class RWKV7ForCausalLM(RWKV7PreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RWKV7Model(config)
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
        past_key_values: Optional[Cache] = None,
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
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[torch.LongTensor] = None,
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
        has_labels = (labels is not None) or (shift_labels is not None)
        if not (fuse_linear_and_cross_entropy and has_labels):
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if has_labels:
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion

            # shift_labels: See https://github.com/huggingface/transformers/pull/36607/files.
            if shift_labels is None:
                shift_labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            shift_labels = shift_labels.to(hidden_states.device)

            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, shift_labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(shift_labels.numel(), -1), shift_labels.view(-1))

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
