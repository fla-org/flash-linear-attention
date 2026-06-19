# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.nsa.configuration_nsa import NSAConfig
from fla.models.nsa.modeling_nsa import NSAForCausalLM, NSAModel

AutoConfig.register(NSAConfig.model_type, NSAConfig, exist_ok=True)
AutoModel.register(NSAConfig, NSAModel, exist_ok=True)
AutoModelForCausalLM.register(NSAConfig, NSAForCausalLM, exist_ok=True)


__all__ = ['NSAConfig', 'NSAForCausalLM', 'NSAModel']
