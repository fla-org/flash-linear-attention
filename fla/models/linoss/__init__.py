from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.linoss.configuration_linoss import LinOSSConfig
from fla.models.linoss.modeling_linoss import LinOSSForCausalLM, LinOSSModel

AutoConfig.register(LinOSSConfig.model_type, LinOSSConfig, exist_ok=True)
AutoModel.register(LinOSSConfig, LinOSSModel, exist_ok=True)
AutoModelForCausalLM.register(LinOSSConfig, LinOSSForCausalLM, exist_ok=True)


__all__ = ['LinOSSConfig', 'LinOSSForCausalLM', 'LinOSSModel']
