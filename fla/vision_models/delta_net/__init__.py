from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.vision_models.delta_net.configuration_delta_net import DeltaNetVisionConfig
from fla.vision_models.delta_net.modeling_delta_net import DeltaNetForImageClassification, DeltaNetVisionModel, DeltaNetForMaskedImageModeling

AutoConfig.register(DeltaNetVisionConfig.model_type, DeltaNetVisionConfig)
AutoModelForImageClassification.register(DeltaNetVisionConfig, DeltaNetForImageClassification)
AutoModelForMaskedImageModeling.register(DeltaNetVisionConfig, DeltaNetForMaskedImageModeling)
AutoModel.register(DeltaNetVisionConfig, DeltaNetVisionModel)

__all__ = [
    "DeltaNetVisionConfig",
    "DeltaNetForImageClassification",
    "DeltaNetVisionModel",
    "DeltaNetForMaskedImageModeling"
]
