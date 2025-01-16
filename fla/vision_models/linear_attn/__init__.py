from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.linear_attn.configuration_linear_attn import LinearAttentionVisionConfig
from fla.vision_models.linear_attn.modeling_linear_attn import LinearAttentionForImageClassification

AutoConfig.register(LinearAttentionVisionConfig.model_type, LinearAttentionVisionConfig)
AutoModelForImageClassification.register(LinearAttentionVisionConfig, LinearAttentionForImageClassification)

__all__ = [
    'LinearAttentionVisionConfig',
    'LinearAttentionForImageClassification'
]
