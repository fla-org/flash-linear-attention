from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.hgrn2.configuration_hgrn2 import HGRN2VisionConfig
from fla.vision_models.hgrn2.modeling_hgrn2 import HGRN2ForImageClassification

AutoConfig.register(HGRN2VisionConfig.model_type, HGRN2VisionConfig)
AutoModelForImageClassification.register(HGRN2VisionConfig, HGRN2ForImageClassification)

__all__ = [
    'HGRN2VisionConfig',
    'HGRN2ForImageClassification'
]
