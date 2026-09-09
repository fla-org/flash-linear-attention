# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.precond_kda.configuration_precond_kda import PrecondKDAConfig
from fla.models.precond_kda.modeling_precond_kda import PrecondKDAForCausalLM, PrecondKDAModel

AutoConfig.register(PrecondKDAConfig.model_type, PrecondKDAConfig, exist_ok=True)
AutoModel.register(PrecondKDAConfig, PrecondKDAModel, exist_ok=True)
AutoModelForCausalLM.register(PrecondKDAConfig, PrecondKDAForCausalLM, exist_ok=True)

__all__ = ['PrecondKDAConfig', 'PrecondKDAForCausalLM', 'PrecondKDAModel']
