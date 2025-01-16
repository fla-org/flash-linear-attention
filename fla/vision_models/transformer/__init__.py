from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.transformer.configuration_transformer import TransformerVisionConfig
from fla.vision_models.transformer.modeling_transformer import TransformerForImageClassification

AutoConfig.register(TransformerVisionConfig.model_type, TransformerVisionConfig)
AutoModelForImageClassification.register(TransformerVisionConfig, TransformerForImageClassification)

__all__ = [
    'TransformerVisionConfig',
    'TransformerForImageClassification'
]
