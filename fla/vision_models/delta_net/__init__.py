from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.delta_net.configuration_delta_net import DeltaNetVisionConfig
from fla.vision_models.delta_net.modeling_delta_net import DeltaNetForImageClassification

# Register the model with transformers
AutoConfig.register("delta_net_vision", DeltaNetVisionConfig)
AutoModelForImageClassification.register(DeltaNetVisionConfig, DeltaNetForImageClassification)

__all__ = [
    'DeltaNetVisionConfig',
    'DeltaNetForImageClassification'
]
