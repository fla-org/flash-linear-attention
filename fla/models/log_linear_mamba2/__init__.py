
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.log_linear_mamba2.configuration_log_linear_mamba2 import LogLinearMamba2Config
from fla.models.log_linear_mamba2.modeling_log_linear_mamba2 import (
    LogLinearMamba2Block,
    LogLinearMamba2ForCausalLM,
    LogLinearMamba2Model,
)

AutoConfig.register(LogLinearMamba2Config.model_type, LogLinearMamba2Config, exist_ok=True)
AutoModel.register(LogLinearMamba2Config, LogLinearMamba2Model, exist_ok=True)
AutoModelForCausalLM.register(LogLinearMamba2Config, LogLinearMamba2ForCausalLM, exist_ok=True)


__all__ = ['LogLinearMamba2Config', 'LogLinearMamba2ForCausalLM', 'LogLinearMamba2Model', 'LogLinearMamba2Block']
