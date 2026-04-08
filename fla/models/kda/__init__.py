# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.kda.configuration_kda import KDAConfig
from fla.models.kda.modeling_kda import KDAForCausalLM, KDAModel

AutoConfig.register(KDAConfig.model_type, KDAConfig, exist_ok=True)
AutoModel.register(KDAConfig, KDAModel, exist_ok=True)
AutoModelForCausalLM.register(KDAConfig, KDAForCausalLM, exist_ok=True)

__all__ = ['KDAConfig', 'KDAForCausalLM', 'KDAModel']
