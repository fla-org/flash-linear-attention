# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gsa.configuration_gsa import GSAConfig
from fla.models.gsa.modeling_gsa import GSAForCausalLM, GSAModel

AutoConfig.register(GSAConfig.model_type, GSAConfig, exist_ok=True)
AutoModel.register(GSAConfig, GSAModel, exist_ok=True)
AutoModelForCausalLM.register(GSAConfig, GSAForCausalLM, exist_ok=True)


__all__ = ['GSAConfig', 'GSAForCausalLM', 'GSAModel']
