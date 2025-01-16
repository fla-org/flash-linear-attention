from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.gla.configuration_gla import GLAVisionConfig
from fla.vision_models.gla.modeling_gla import GLAForImageClassification

AutoConfig.register(GLAVisionConfig.model_type, GLAVisionConfig)
AutoModelForImageClassification.register(GLAVisionConfig, GLAForImageClassification)

__all__ = [
    'GLAVisionConfig',
    'GLAForImageClassification'
]
