from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gated_deltaproduct.configuration_gated_deltaproduct import GatedDeltaProductConfig
from fla.models.gated_deltaproduct.modeling_gated_deltaproduct import GatedDeltaProductForCausalLM, GatedDeltaProductModel

AutoConfig.register(GatedDeltaProductConfig.model_type, GatedDeltaProductConfig)
AutoModel.register(GatedDeltaProductConfig, GatedDeltaProductModel)
AutoModelForCausalLM.register(GatedDeltaProductConfig, GatedDeltaProductForCausalLM)

__all__ = [
    "GatedDeltaProductConfig",
    "GatedDeltaProductForCausalLM",
    "GatedDeltaProductModel",
]
