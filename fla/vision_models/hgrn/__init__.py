from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.hgrn.configuration_hgrn import HGRNVisionConfig
from fla.vision_models.hgrn.modeling_hgrn import HGRNForImageClassification

AutoConfig.register(HGRNVisionConfig.model_type, HGRNVisionConfig)
AutoModelForImageClassification.register(HGRNVisionConfig, HGRNForImageClassification)

__all__ = [
    'HGRNVisionConfig',
    'HGRNForImageClassification'
]
