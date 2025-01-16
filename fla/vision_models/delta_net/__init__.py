from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.delta_net.configuration_delta_net import DeltaNetVisionConfig
from fla.vision_models.delta_net.modeling_delta_net import DeltaNetForImageClassification

AutoConfig.register(DeltaNetVisionConfig.model_type, DeltaNetVisionConfig)
AutoModelForImageClassification.register(DeltaNetVisionConfig, DeltaNetForImageClassification)

__all__ = [
    'DeltaNetVisionConfig',
    'DeltaNetForImageClassification'
]
