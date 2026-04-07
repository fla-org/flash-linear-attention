# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mamba2.configuration_mamba2 import Mamba2Config
from fla.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM, Mamba2Model

AutoConfig.register(Mamba2Config.model_type, Mamba2Config, exist_ok=True)
AutoModel.register(Mamba2Config, Mamba2Model, exist_ok=True)
AutoModelForCausalLM.register(Mamba2Config, Mamba2ForCausalLM, exist_ok=True)


__all__ = ['Mamba2Config', 'Mamba2ForCausalLM', 'Mamba2Model']
