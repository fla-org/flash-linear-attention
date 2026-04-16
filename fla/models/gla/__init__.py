# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gla.configuration_gla import GLAConfig
from fla.models.gla.modeling_gla import GLAForCausalLM, GLAModel

AutoConfig.register(GLAConfig.model_type, GLAConfig, exist_ok=True)
AutoModel.register(GLAConfig, GLAModel, exist_ok=True)
AutoModelForCausalLM.register(GLAConfig, GLAForCausalLM, exist_ok=True)


__all__ = ['GLAConfig', 'GLAForCausalLM', 'GLAModel']
