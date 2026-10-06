# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
import re
import warnings
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from packaging import version
from transformers import __version__ as transformers_version
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.layers.gla import GatedLinearAttention
from fla.models.gla.configuration_gla import GLAConfig
from fla.models.hybrid import get_hybrid_attention_spec
from fla.models.utils import Cache, FLAUnsupportedCacheGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as GLAMLP
from fla.modules.l2warp import l2_warp
from fla.modules.residuals import BaseResidual, get_residual_class

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


def _legacy_residual_keys(num_layers: int) -> dict[str, str]:
    """Map official and external-attn-norm checkpoints to residual-owned parameters."""
    keys = {}
    for i in range(num_layers):
        previous = f'layers.{i - 1}.residual_mlp' if i else 'layers.0.residual_attn'
        input_norm = f'{previous}.norm.weight' if i else f'{previous}.input_norm.weight'
        keys[f'layers.{i}.attn_norm.weight'] = input_norm
        keys[f'attn_norms.{i}.weight'] = input_norm
        keys[f'layers.{i}.mlp_norm.weight'] = f'layers.{i}.residual_attn.norm.weight'
        for old, new in [('res_proj', 'query'), ('res_norm', 'key_norm')]:
            keys[f'layers.{i}.attn_{old}.weight'] = f'{previous}.{new if i else "input_" + new}.weight'
            keys[f'layers.{i}.mlp_{old}.weight'] = f'layers.{i}.residual_attn.{new}.weight'
    keys['norm.weight'] = f'layers.{num_layers - 1}.residual_mlp.norm.weight'
    keys['res_proj.weight'] = f'layers.{num_layers - 1}.residual_mlp.query.weight'
    keys['res_norm.weight'] = f'layers.{num_layers - 1}.residual_mlp.key_norm.weight'
    return keys


class GLABlock(GradientCheckpointingLayer):

    def __init__(self, config: GLAConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        attn_spec = get_hybrid_attention_spec(config.attn, layer_idx=layer_idx)
        if attn_spec is not None:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=attn_spec['num_heads'],
                num_kv_heads=attn_spec['num_kv_heads'],
                qkv_bias=attn_spec['qkv_bias'],
                window_size=attn_spec['window_size'],
                rope_theta=attn_spec['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
            )
        else:
            self.attn = GatedLinearAttention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                feature_map=config.feature_map,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                use_output_gate=config.use_output_gate,
                gate_fn=config.hidden_act,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                clamp_min=config.clamp_min,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
            )

        self.mlp = GLAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        residual_cls = get_residual_class(config.residual_mode)
        self.residual_attn = residual_cls(**config.get_residual_kwargs(sub_layer_idx=layer_idx * 2))
        self.residual_mlp = residual_cls(**config.get_residual_kwargs(sub_layer_idx=layer_idx * 2 + 1))
        for residual in (self.residual_attn, self.residual_mlp):
            for module in residual.modules():
                if module is not residual:
                    module._is_residual_child = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        history: Any = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None, Any]:
        if history is None:
            hidden_states, history = self.residual_attn.initialize(hidden_states)

        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states, history = self.residual_attn(hidden_states, history)

        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states, history = self.residual_mlp(hidden_states, history)
        outputs = (hidden_states, attentions, past_key_values, history)
        return outputs


class GLAPreTrainedModel(PreTrainedModel):

    config_class = GLAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GLABlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # newer HF loaders bypass module load hooks, including when loading onto the meta device.
        if version.parse(transformers_version) >= version.parse('4.51.0'):
            config = kwargs.get('config')
            if not isinstance(config, GLAConfig):
                config = cls.config_class.from_pretrained(config or pretrained_model_name_or_path, **{
                    key: kwargs[key] for key in ('cache_dir', 'revision', 'token', 'local_files_only', 'subfolder')
                    if key in kwargs
                })
            key_mapping = dict(kwargs.pop('key_mapping', None) or {})
            if config.residual_mode != 'mhc':
                for old_key, new_key in _legacy_residual_keys(config.num_hidden_layers).items():
                    key_mapping.setdefault(r'^(model\.|)' + re.escape(old_key) + '$', r'\1' + new_key)
            kwargs['key_mapping'] = key_mapping
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # residual history carries autograd edges independently of the normalized positional input.
        checkpoint_kwargs = {'use_reentrant': False, **(gradient_checkpointing_kwargs or {})}
        if checkpoint_kwargs['use_reentrant']:
            raise ValueError('GLA residual history requires gradient checkpointing with use_reentrant=False')
        return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)

    def _initialize_weights(self, module: nn.Module, *args, **kwargs):
        if getattr(module, '_is_residual_child', False):
            return
        if isinstance(module, BaseResidual):
            # a residual owns initialization of descendants even when its direct parameters were loaded.
            if not all(getattr(child, '_is_hf_initialized', False) for child in module.modules()):
                self._init_weights(module)
                for child in module.modules():
                    child._is_hf_initialized = True
            return
        super()._initialize_weights(module, *args, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, BaseResidual):
            module.reset_parameters()
            return
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if getattr(module, '_is_attnres_proj', False):
                nn.init.zeros_(module.weight)
            else:
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py
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


class GLAModel(GLAPreTrainedModel):

    def __init__(self, config: GLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GLABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False

        self.post_init()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.config.residual_mode != 'mhc':
            for old_key, new_key in _legacy_residual_keys(len(self.layers)).items():
                old_key, new_key = prefix + old_key, prefix + new_key
                if old_key in state_dict and new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            warnings.warn("`GLAModel` does not `output_attentions` now, setting it to `False`.")
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

        history = None

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, attentions, past_key_values, history = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                history=history,
                **kwargs,
            )

            if output_attentions:
                all_attns += (attentions,)

            if output_hidden_states:
                # Intermediate entries report residual states; the final entry is the normalized model output.
                all_hidden_states += (
                    hidden_states if layer_idx == len(self.layers) - 1 else layer.residual_mlp.get_hidden_state(history),
                )

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class GLAForCausalLM(GLAPreTrainedModel, FLAUnsupportedCacheGenerationMixin):

    # transformers 5 requires target-to-source mappings, while 4.x uses a list of tied keys.
    _tied_weights_keys = (
        {"lm_head.weight": "model.embeddings.weight"}
        if hasattr(PreTrainedModel, 'get_expanded_tied_weights_keys')
        else ["lm_head.weight"]
    )

    def __init__(self, config):
        super().__init__(config)
        self.model = GLAModel(config)
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
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast:
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
            **kwargs,
        )

        hidden_states = outputs[0]

        loss, logits = None, None
        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(
                hidden_states
                if labels is not None or logits_to_keep is None
                else hidden_states[:, -logits_to_keep:]
            )
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
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
