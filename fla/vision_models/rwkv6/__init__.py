from transformers import AutoConfig, AutoModelForImageClassification

from fla.vision_models.rwkv6.configuration_rwkv6 import RWKV6VisionConfig
from fla.vision_models.rwkv6.modeling_rwkv6 import RWKV6ForImageClassification

AutoConfig.register(RWKV6VisionConfig.model_type, RWKV6VisionConfig)
AutoModelForImageClassification.register(RWKV6VisionConfig, RWKV6ForImageClassification)

__all__ = [
    'RWKV6VisionConfig',
    'RWKV6ForImageClassification'
]
