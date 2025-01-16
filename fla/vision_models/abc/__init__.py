from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.abc.configuration_abc import ABCVisionConfig
from fla.vision_models.abc.modeling_abc import ABCForImageClassification

AutoConfig.register(ABCVisionConfig.model_type, ABCVisionConfig)
AutoModelForImageClassification.register(ABCVisionConfig, ABCForImageClassification)

__all__ = [
    'ABCVisionConfig',
    'ABCForImageClassification'
]
