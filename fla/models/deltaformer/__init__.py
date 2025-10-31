
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.deltaformer.configuration_deltaformer import DeltaFormerConfig
from fla.models.deltaformer.modeling_deltaformer import DeltaFormerForCausalLM, DeltaFormerModel

AutoConfig.register(DeltaFormerConfig.model_type, DeltaFormerConfig, exist_ok=True)
AutoModel.register(DeltaFormerConfig, DeltaFormerModel, exist_ok=True)
AutoModelForCausalLM.register(DeltaFormerConfig, DeltaFormerForCausalLM, exist_ok=True)

__all__ = ['DeltaFormerConfig', 'DeltaFormerForCausalLM', 'DeltaFormerModel']
