from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetVisionConfig
from fla.vision_models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForImageClassification

AutoConfig.register(GatedDeltaNetVisionConfig.model_type, GatedDeltaNetVisionConfig)
AutoModelForImageClassification.register(GatedDeltaNetVisionConfig, GatedDeltaNetForImageClassification)

__all__ = [
    'GatedDeltaNetVisionConfig',
    'GatedDeltaNetForImageClassification'
]

