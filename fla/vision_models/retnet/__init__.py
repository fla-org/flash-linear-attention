from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.retnet.configuration_retnet import RetNetVisionConfig
from fla.vision_models.retnet.modeling_retnet import RetNetForImageClassification

AutoConfig.register(RetNetVisionConfig.model_type, RetNetVisionConfig)
AutoModelForImageClassification.register(RetNetVisionConfig, RetNetForImageClassification)

__all__ = [
    'RetNetVisionConfig',
    'RetNetForImageClassification'
]
