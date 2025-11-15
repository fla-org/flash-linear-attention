
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.stickbreaking_attn.configuration_stickbreaking_attn import StickBreakingAttentionConfig
from fla.models.stickbreaking_attn.modeling_stickbreaking_attn import (
    StickBreakingAttentionForCausalLM,
    StickBreakingAttentionModel,
)

AutoConfig.register(StickBreakingAttentionConfig.model_type, StickBreakingAttentionConfig, exist_ok=True)
AutoModel.register(StickBreakingAttentionConfig, StickBreakingAttentionModel, exist_ok=True)
AutoModelForCausalLM.register(StickBreakingAttentionConfig, StickBreakingAttentionForCausalLM, exist_ok=True)


__all__ = ['StickBreakingAttentionConfig', 'StickBreakingAttentionForCausalLM', 'StickBreakingAttentionModel']
