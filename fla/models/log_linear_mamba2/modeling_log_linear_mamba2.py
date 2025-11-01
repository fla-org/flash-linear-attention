import math
import warnings

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.log_linear_mamba2 import LogLinearAttentionState, LogLinearMamba2
from fla.models.log_linear_mamba2.configuration_log_linear_mamba2 import LogLinearMamba2Config
from fla.models.mamba2.modeling_mamba2 import Mamba2Cache, Mamba2CausalLMOutput, Mamba2Output
from fla.models.utils import FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, GatedMLP, RMSNorm

logger = logging.get_logger(__name__)


class LogLinearMamba2Cache(Mamba2Cache):
    def __init__(
        self,
        config: LogLinearMamba2Config,
        batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: str | None = None,
    ):
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.n_groups = config.n_groups
        self.state_size = config.state_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.intermediate_size = int(config.expand * config.hidden_size)

        self.conv_states = {
            i: torch.zeros(
                batch_size,
                self.intermediate_size + 2 * config.n_groups * config.state_size,
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            )
            for i in range(config.num_hidden_layers)
        }
        self.hssm_states = dict.fromkeys(range(config.num_hidden_layers))

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False,
    ) -> torch.Tensor:
        if new_conv_state.dtype != self.conv_states[layer_idx].dtype:
            warnings.warn(
                f"`new_conv_state.dtype` ({new_conv_state.dtype}) does not match the cache's dtype "
                f"({self.conv_states[layer_idx].dtype}), casting.",
                stacklevel=2,
            )
            new_conv_state = new_conv_state.to(dtype=self.conv_states[layer_idx].dtype)

        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(
                self.conv_states[layer_idx].device,
            )
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(
                shifts=-1, dims=-1,
            )
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(
                self.conv_states[layer_idx].device,
            )
        return self.conv_states[layer_idx]

    def update_hssm_state(
        self, layer_idx: int, new_hssm_state: LogLinearAttentionState,
    ) -> LogLinearAttentionState:
        self.hssm_states[layer_idx] = new_hssm_state
        return self.hssm_states[layer_idx]

    def reset(self) -> None:
        for k in self.conv_states.keys():
            self.conv_states[k].zero_()
        for k in self.hssm_states.keys():
            self.hssm_states[k].reset_states()


class LogLinearMamba2Block(nn.Module):
    def __init__(self, config: LogLinearMamba2Config, layer_idx: int) -> None:
        super().__init__()
        if config.residual_in_fp32:
            raise NotImplementedError
        self.config = config
        self.layer_idx = layer_idx
        self.mixer_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
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
        hidden_states,
        cache_params: LogLinearMamba2Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.mixer_norm(hidden_states)
        hidden_states = self.mixer(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
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
        return hidden_states


class LogLinearMamba2PreTrainedModel(PreTrainedModel, FLAGenerationMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LogLinearMamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["LogLinearMamba2Block"]
    supports_gradient_checkpointing = True
    _is_stateful = True

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
        self.norm_f = RMSNorm(config.hidden_size, eps=config.norm_eps)
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
        inputs_embeds: torch.LongTensor | None = None,
        cache_params: LogLinearMamba2Cache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | Mamba2Output:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds",
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = LogLinearMamba2Cache(
                    self.config,
                    inputs_embeds.size(0),
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                cache_position = torch.arange(
                    0, self.config.conv_kernel, device=inputs_embeds.device,
                )
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically",
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__,
                    hidden_states,
                    cache_params,
                    cache_position,
                    attention_mask,
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, cache_params, all_hidden_states]
                if v is not None
            )

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
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
        inputs_embeds: torch.FloatTensor | None = None,
        cache_params: LogLinearMamba2Cache | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int | None = 0,
        **kwargs,  # for now we need this for generation
    ) -> tuple | Mamba2CausalLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
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

        return Mamba2CausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
        )
