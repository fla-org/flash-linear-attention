# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.hgrn2.configuration_hgrn2 import HGRN2Config
from fla.models.hgrn2.modeling_hgrn2 import HGRN2ForCausalLM, HGRN2Model

AutoConfig.register(HGRN2Config.model_type, HGRN2Config, exist_ok=True)
AutoModel.register(HGRN2Config, HGRN2Model, exist_ok=True)
AutoModelForCausalLM.register(HGRN2Config, HGRN2ForCausalLM, exist_ok=True)


__all__ = ['HGRN2Config', 'HGRN2ForCausalLM', 'HGRN2Model']
