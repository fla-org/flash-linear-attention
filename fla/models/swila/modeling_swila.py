# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""
Switching Linear Attention Model Implementation

This module implements the full SwiLA model architecture following the
flash-linear-attention design patterns, compatible with HuggingFace transformers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers import SwitchingLinearAttention
from fla.models.cache_utils import Cache
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.modeling_utils import FLAPreTrainedModel, get_attention
from fla.models.swila.configuration_swila import SwiLAConfig
from fla.modules import GatedMLP as SWILAMLP
from fla.modules import RMSNorm
from fla.modules.swila import initialization

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class SwiLABlock(GradientCheckpointingLayer):
    """
    Single SwiLA decoder block.

    Each block consists of:
    1. Pre-norm + Switching Linear Attention + Residual
    2. Pre-norm + MLP + Residual
    """

    def __init__(self, config: SwiLAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention components
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = get_attention(config=config, layer_idx=layer_idx)
        self.is_transformer_attention = self.attn is not None
        if not self.is_transformer_attention:
            self.attn = SwitchingLinearAttention(
                mode=config.mode,
                d_model=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_experts=config.num_experts,
                architecture_profile=getattr(config, 'architecture_profile', 'current'),
                use_gate=config.use_gate,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                conv_bias=config.conv_bias,
                layer_idx=layer_idx,
                qk_norm=config.qk_norm,
                norm_eps=config.norm_eps,
                scale=config.scale,
                # SwiLABlock consumes the full four-value layer contract:
                # hidden states, attentions, cache, and auxiliary loss.
                return_output_only=False,
                use_pi_conv=config.use_pi_conv,
                pi_proj_rank=config.pi_proj_rank,
                b_proj_rank=config.b_proj_rank,
                low_rank_proj_type=getattr(config, 'low_rank_proj_type', 'swiglu'),
                pi_low_rank_proj_type=getattr(config, 'pi_low_rank_proj_type', None),
                b_low_rank_proj_type=getattr(config, 'b_low_rank_proj_type', 'linear'),
                alpha_low_rank_proj_type=getattr(config, 'alpha_low_rank_proj_type', 'linear'),

                init_state_mode=config.init_state_mode,
                temporal=config.temporal,


                inject_noise=config.inject_noise,
                inject_gate_noise=config.inject_gate_noise,

                pi_init_logit_std=getattr(config, 'pi_init_logit_std', 0.1),
                pi_align_qk_init=getattr(config, 'pi_align_qk_init', True),
                pi_noise_init_std=getattr(config, 'pi_noise_init_std', 0.02),
                gate_init_bias=config.gate_init_bias,
                gate_noise_init_std=getattr(
                    config,
                    'gate_noise_init_std',
                    0.02,
                ),
                mi_loss_weight=config.mi_loss_weight,

                use_forget_gate=getattr(config, 'use_forget_gate', False),
                alpha_proj_rank=getattr(config, 'alpha_proj_rank', 16),
            )

        # MLP components
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = SWILAMLP(
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
        **kwargs: Unpack[dict],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | dict[str, torch.Tensor] | None,
        Cache | list[torch.FloatTensor] | None,
        torch.Tensor | None,
    ]:
        """Return hidden states, optional attentions, cache, and auxiliary loss."""
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        if self.is_transformer_attention:
            hidden_states, attentions, past_key_values = self.attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
            aux_loss = hidden_states.new_zeros(())
        else:
            hidden_states, attentions, past_key_values, aux_loss = self.attn(
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

        outputs = (hidden_states, attentions, past_key_values, aux_loss)

        return outputs


class SwiLAPreTrainedModel(FLAPreTrainedModel):
    """
    Base class for SwiLA models providing weight initialization and other utilities.
    """

    config_class = SwiLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SwiLABlock"]
    _supports_cache_class = True

    @classmethod
    def from_pretrained(cls, *model_args, **kwargs):
        """Finish HF loading, then restore SwiLA-only metadata and defaults.

        Transformers' optimized ``from_pretrained`` loader does not call
        ``nn.Module.load_state_dict`` and therefore bypasses module load hooks.
        The public loading-info return contract preserves the complete base
        lifecycle and the caller's requested return shape.
        Only Python-only optimizer markers and temporal-noise biases absent
        from legacy checkpoints are repaired; loaded values are not reset.
        """
        output_loading_info = kwargs.pop("output_loading_info", False)
        model, loading_info = super().from_pretrained(
            *model_args,
            output_loading_info=True,
            **kwargs,
        )
        missing_keys = set(loading_info["missing_keys"])
        with torch.no_grad():
            for module_name, layer in model.named_modules():
                if not isinstance(layer, SwitchingLinearAttention):
                    continue
                initialization.restore_parameter_metadata(layer)
                if not layer.temporal or not layer.inject_gate_noise:
                    continue
                prefix = f"{module_name}." if module_name else ""
                for suffix, projection in (
                    ("W_noise_gate_kv.bias", layer.W_noise_gate_kv),
                    ("W_noise_gate_q.bias", layer.W_noise_gate_q),
                ):
                    if f"{prefix}{suffix}" in missing_keys:
                        initialization.initialize_missing_gate_noise_bias(layer, projection)
        return (model, loading_info) if output_loading_info else model

    def post_init(self):
        """Initialize generic weights, then restore SwiLA-specific initialization.

        FineWeb/Flame constructs the model under ``torch.device("meta")``,
        applies parallelism, materializes storage with ``to_empty()``, and
        calls ``post_init()`` again. Hugging Face ``from_pretrained()`` instead
        constructs empty weights by routing registered parameters to ``meta``
        before loading checkpoint values. These intentional all-meta
        constructor passes are the only phases allowed to remain on meta.
        Every later pass explicitly restores SwiLA initialization after generic
        leaf initialization and fails closed if any SwiLA parameter remains
        unmaterialized.
        """
        # HF consults this plan before loading checkpoint tensors.
        # Restoring dtype after loading
        # cannot undo a BF16/FP16 rounding of the stored decay parameters.
        # Leave it empty for models without SwiLA decay parameters, including
        # Transformer-only hybrid configurations.
        self._keep_in_fp32_modules_strict = (
            ["A_log", "dt_bias"]
            if any(
                isinstance(module, SwitchingLinearAttention) and module.use_forget_gate
                for module in self.modules()
            )
            else []
        )
        super().post_init()

        swila_layers = [
            module
            for module in self.modules()
            if isinstance(module, SwitchingLinearAttention)
        ]
        meta_parameters = [
            parameter
            for layer in swila_layers
            for parameter in layer.parameters()
            if parameter.is_meta
        ]
        if meta_parameters:
            all_parameters_are_meta = all(
                parameter.is_meta for parameter in self.parameters()
            )
            # ``torch.get_default_device()`` does not report an active
            # ``with torch.device("meta")`` context on every supported
            # PyTorch release.  A factory probe does: it follows both an
            # active device context and a process-wide default device.
            #
            # Transformers 4.57 ``from_pretrained`` uses a different
            # mechanism: its empty-weight context monkeypatches
            # ``nn.Module.register_parameter`` to route parameters to meta
            # while ordinary factories still use CPU. Registering a
            # disposable parameter detects that context without depending on
            # private Transformers or Accelerate state. Outside either
            # construction context both probes are real, so an accidentally
            # unmaterialized model still fails closed.
            active_construction_device = torch.empty(0).device
            registration_probe = nn.Module()
            registration_probe.register_parameter(
                "_swila_meta_probe",
                nn.Parameter(torch.empty(0), requires_grad=False),
            )
            active_registration_device = (
                registration_probe._swila_meta_probe.device
            )
            if (
                all_parameters_are_meta
                and (
                    active_construction_device.type == "meta"
                    or active_registration_device.type == "meta"
                )
            ):
                if getattr(self.config, "architecture_profile", "current") == "historical_8dc05f93":
                    self._historical_meta_materialization_pending = True
                return
            raise RuntimeError(
                "SwiLA post_init found unmaterialized meta parameters outside "
                "the intentional meta-device construction pass"
            )

        historical_materialization = self.__dict__.pop("_historical_meta_materialization_pending", False)
        for layer in swila_layers:
            if historical_materialization and layer.architecture_profile == "historical_8dc05f93":
                # Historical Flame materialization ran only HF's bottom-up
                # reset, unlike real-device model construction's explicit final
                # reset. Preserve those RNG draws, but retain the learned-state
                # materialization fix (old to_empty left that storage undefined).
                initialization.initialize_state_parameters(layer)
                continue
            # This is the authoritative last writer. It also makes the
            # lifecycle independent of whether a particular Transformers
            # version visits composite modules during generic initialization.
            layer.reset_parameters()

    def _init_weights(self, module: nn.Module):
        """Initialize generic model weights, then dispatch custom module resets.

        Linear/Conv/Embedding leaves receive ``initializer_range`` initialization.
        During the bottom-up Hugging Face traversal, composite modules such as
        ``SwitchingLinearAttention`` are visited afterward and can restore their
        specialized pi, beta, alpha, temporal-gate, and noise initialization via
        ``reset_parameters()``.
        """
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            if not getattr(module.weight, "_is_hf_initialized", False):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.initializer_range,
                )
            if (
                module.bias is not None
                and not getattr(module.bias, "_is_hf_initialized", False)
            ):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if not getattr(module.weight, "_is_hf_initialized", False):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.initializer_range,
                )
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()


class SwiLAModel(SwiLAPreTrainedModel):
    """
    SwiLA Model outputting raw hidden states.

    This is the base transformer model without a language modeling head.
    """

    def __init__(self, config: SwiLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([SwiLABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

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
        **kwargs: Unpack[dict],
    ) -> tuple | BaseModelOutputWithPast:
        """Run all decoder blocks and preserve their optional attention values.

        When ``output_attentions=True``, ``attentions`` contains one entry per
        decoder block. Each entry is exactly the attention/responsibility value
        returned by that block's attention implementation.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.return_dict

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

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        total_aux_loss = 0.0
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attentions, past_key_values, aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        # Store accumulated auxiliary loss for SwiLAForCausalLM to consume.
        # This stays inside the autograd graph (not checkpointed at this level),
        # so MI loss gradients flow correctly during backward.
        self._total_aux_loss = total_aux_loss

        # add hidden states from the last decoder layer
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


class SwiLAForCausalLM(SwiLAPreTrainedModel, FLAUnsupportedCacheGenerationMixin):

    # Trainer must average CE and sequence-mean MI together across microbatches.
    accepts_loss_kwargs = False
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = SwiLAModel(config)
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
        logits_to_keep: int | torch.Tensor | None = 0,
        num_items_in_batch: int | torch.Tensor | None = None,
        shift_labels: torch.LongTensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast:
        """Return CE-valued loss with connected MI gradients and decoder attentions.

        ``shift_labels`` and ``num_items_in_batch`` follow FLA's CE conventions.
        MI retains its configured per-sequence/layer normalization independently
        of an explicitly supplied CE token-count denominator; custom callers own
        that accumulation scaling. Default Trainer averages the complete per-call
        CE+MI objective instead of injecting an accumulated token denominator.
        Labeled unfused calls return all logits.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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
            shift_labels=shift_labels,
            num_items_in_batch=num_items_in_batch,
        )
        if loss is not None:
            # keep MI connected while reporting the unmodified CE value
            total_aux_loss = getattr(self.model, '_total_aux_loss', 0.0)
            if isinstance(total_aux_loss, torch.Tensor):
                loss = loss + (total_aux_loss - total_aux_loss.detach())

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
