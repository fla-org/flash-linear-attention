
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mom.configuration_mom import MomConfig
from fla.models.mom.modeling_mom import MomForCausalLM, MomModel

AutoConfig.register(MomConfig.model_type, MomConfig, exist_ok=True)
AutoModel.register(MomConfig, MomModel, exist_ok=True)
AutoModelForCausalLM.register(MomConfig, MomForCausalLM, exist_ok=True)

__all__ = ['MomConfig', 'MomForCausalLM', 'MomModel']
