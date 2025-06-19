# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mamba.configuration_mamba import MambaConfig
from fla.models.mamba.modeling_mamba import MambaBlock, MambaForCausalLM, MambaModel

AutoConfig.register(MambaConfig.model_type, MambaConfig, exist_ok=True)
AutoModel.register(MambaConfig, MambaModel, exist_ok=True)
AutoModelForCausalLM.register(MambaConfig, MambaForCausalLM, exist_ok=True)


__all__ = ['MambaConfig', 'MambaForCausalLM', 'MambaModel', 'MambaBlock']
