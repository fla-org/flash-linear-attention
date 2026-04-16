# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mom.configuration_mom import MomConfig
from fla.models.mom.modeling_mom import MomForCausalLM, MomModel

AutoConfig.register(MomConfig.model_type, MomConfig, exist_ok=True)
AutoModel.register(MomConfig, MomModel, exist_ok=True)
AutoModelForCausalLM.register(MomConfig, MomForCausalLM, exist_ok=True)

__all__ = ['MomConfig', 'MomForCausalLM', 'MomModel']
