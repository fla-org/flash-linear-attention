# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gsa2.configuration_gsa2 import GSA2Config
from fla.models.gsa2.modeling_gsa2 import GSA2ForCausalLM, GSA2Model

AutoConfig.register(GSA2Config.model_type, GSA2Config, exist_ok=True)
AutoModel.register(GSA2Config, GSA2Model, exist_ok=True)
AutoModelForCausalLM.register(GSA2Config, GSA2ForCausalLM, exist_ok=True)

__all__ = ['GSA2Config', 'GSA2ForCausalLM', 'GSA2Model']
