# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.log_linear_mamba2.configuration_log_linear_mamba2 import LogLinearMamba2Config
from fla.models.log_linear_mamba2.modeling_log_linear_mamba2 import LogLinearMamba2ForCausalLM, LogLinearMamba2Model

AutoConfig.register(LogLinearMamba2Config.model_type, LogLinearMamba2Config, exist_ok=True)
AutoModel.register(LogLinearMamba2Config, LogLinearMamba2Model, exist_ok=True)
AutoModelForCausalLM.register(LogLinearMamba2Config, LogLinearMamba2ForCausalLM, exist_ok=True)


__all__ = ['LogLinearMamba2Config', 'LogLinearMamba2ForCausalLM', 'LogLinearMamba2Model']
