# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mla.configuration_mla import MLAConfig
from fla.models.mla.modeling_mla import MLAForCausalLM, MLAModel

AutoConfig.register(MLAConfig.model_type, MLAConfig, exist_ok=True)
AutoModel.register(MLAConfig, MLAModel, exist_ok=True)
AutoModelForCausalLM.register(MLAConfig, MLAForCausalLM, exist_ok=True)


__all__ = ['MLAConfig', 'MLAForCausalLM', 'MLAModel']
