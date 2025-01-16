from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.gsa.configuration_gsa import GSAVisionConfig
from fla.vision_models.gsa.modeling_gsa import GSAForImageClassification

AutoConfig.register(GSAVisionConfig.model_type, GSAVisionConfig)
AutoModelForImageClassification.register(GSAVisionConfig, GSAForImageClassification)

__all__ = [
    'GSAVisionConfig',
    'GSAForImageClassification'
]
