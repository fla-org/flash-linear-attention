import math

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.log_linear_mamba2 import LogLinearMamba2
from fla.models.log_linear_mamba2.configuration_log_linear_mamba2 import LogLinearMamba2Config
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, GatedMLP, RMSNorm

logger = logging.get_logger(__name__)


class LogLinearMamba2Block(nn.Module):
    def __init__(self, config: LogLinearMamba2Config, layer_idx: int) -> None:
        super().__init__()
        if config.residual_in_fp32:
            raise NotImplementedError
        self.config = config
        self.layer_idx = layer_idx
        self.mixer_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=torch.float32)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=torch.float32)
        self.mixer = LogLinearMamba2(
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            hidden_size=config.hidden_size,
            state_size=config.state_size,
            expand=config.expand,
            n_groups=config.n_groups,
            conv_kernel=config.conv_kernel,
            use_conv_bias=config.use_conv_bias,
            hidden_act=config.hidden_act,
            rms_norm=config.rms_norm,
            chunk_size=config.chunk_size,
            time_step_rank=config.time_step_rank,
            time_step_limit=config.time_step_limit,
            time_step_min=config.time_step_min,
            time_step_max=config.time_step_max,
            use_bias=config.use_bias,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
        )
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=4,
            intermediate_size=None,
            hidden_act="swish",
            fuse_swiglu=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.mixer_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(
                hidden_states, residual=residual, prenorm=True,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attentions, past_key_values


class LogLinearMamba2PreTrainedModel(PreTrainedModel, FLAGenerationMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LogLinearMamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["LogLinearMamba2Block"]
    supports_gradient_checkpointing = True
    _supports_cache_class = True

    def _init_weights(
        self,
        module: nn.Module,
        num_residuals_per_layer: int = 2,  # HAttention + MLP
    ):
        """Initialize the weights."""
        if isinstance(module, LogLinearMamba2):
            # --- A_log ---
            A = torch.arange(1, module.num_heads + 1)
            with torch.no_grad():
                if not isinstance(module.A_log, torch.distributed.tensor.DTensor):
                    module.A_log.copy_(torch.log(A))
                else:
                    logger.warning_once("`A_log` is a DTensor, skipping initialization")
            module.A_log._no_weight_decay = True

            # --- D ---
            nn.init.ones_(module.D)
            module.D._no_weight_decay = True

            # --- L ---
            nn.init.ones_(module.L)
            module.L._no_weight_decay = True

            # --- dt_bias ---
            dt = torch.exp(
                torch.rand(self.config.num_heads)
                * (
                    math.log(self.config.time_step_max)
                    - math.log(self.config.time_step_min)
                )
                + math.log(self.config.time_step_min),
            ).clamp(min=self.config.time_step_floor)

            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                if not isinstance(module.dt_bias, torch.distributed.tensor.DTensor):
                    module.dt_bias.copy_(inv_dt)
                else:
                    logger.warning_once(
                        "`dt_bias` is a DTensor, skipping initialization",
                    )
            module.dt_bias._no_reinit = True

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                # guard against deprecated behavior
                if hasattr(module.bias, "_no_reinit"):
                    raise ValueError("This is not supposed to happen")
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, "o_proj"):
                # p = module.o_proj.weight
                # guard against deprecated behavior
                raise ValueError("This is not supposed to happen")
            elif hasattr(module, "out_proj"):
                p = module.out_proj.weight
            elif hasattr(module, "down_proj"):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(
                        num_residuals_per_layer * self.config.num_hidden_layers,
                    )


class LogLinearMamba2Model(LogLinearMamba2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                LogLinearMamba2Block(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ],
        )

        self.gradient_checkpointing = False
        self.norm_f = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=torch.float32)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

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
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        if self.gradient_checkpointing and self.training and (use_cache or past_key_values is not None):
            logger.warning_once("Disabling cache because gradient checkpointing replays the forward pass.")
            use_cache = False
            past_key_values = None
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds",
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for mixer_block in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    mixer_block.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                )
            else:
                hidden_states, attentions, past_key_values = mixer_block(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs,
                )

            if output_attentions and attentions is not None:
                all_attns = all_attns + (attentions,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                i
                for i in [hidden_states, past_key_values, all_hidden_states, all_attns]
                if i is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns if all_attns else None,
        )


class LogLinearMamba2ForCausalLM(LogLinearMamba2PreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.backbone = LogLinearMamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

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

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.backbone(
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
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training

        loss, logits = None, None
        if not fuse_linear_and_cross_entropy or labels is None:
            logits = self.lm_head(
                hidden_states
                if logits_to_keep is None
                else hidden_states[:, -logits_to_keep:],
            )
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (
                    labels[..., 1:],
                    torch.full_like(labels[:, :1], criterion.ignore_index),
                ),
                1,
            )
            if fuse_linear_and_cross_entropy:
                loss = criterion(
                    hidden_states, labels, self.lm_head.weight, self.lm_head.bias,
                )
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

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
