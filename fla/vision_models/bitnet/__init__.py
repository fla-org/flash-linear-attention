from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.bitnet.configuration_bitnet import BitNetVisionConfig
from fla.vision_models.bitnet.modeling_bitnet import BitNetForImageClassification

AutoConfig.register(BitNetVisionConfig, BitNetVisionConfig)
AutoModelForImageClassification.register(BitNetVisionConfig, BitNetForImageClassification)

__all__ = [
    'BitNetVisionConfig',
    'BitNetForImageClassification'
]
