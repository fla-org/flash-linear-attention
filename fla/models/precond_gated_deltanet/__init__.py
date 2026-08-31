# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.precond_gated_deltanet.configuration_precond_gated_deltanet import PrecondGatedDeltaNetConfig
from fla.models.precond_gated_deltanet.modeling_precond_gated_deltanet import (
    PrecondGatedDeltaNetForCausalLM,
    PrecondGatedDeltaNetModel,
)

AutoConfig.register(PrecondGatedDeltaNetConfig.model_type, PrecondGatedDeltaNetConfig, exist_ok=True)
AutoModel.register(PrecondGatedDeltaNetConfig, PrecondGatedDeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(PrecondGatedDeltaNetConfig, PrecondGatedDeltaNetForCausalLM, exist_ok=True)

__all__ = ['PrecondGatedDeltaNetConfig', 'PrecondGatedDeltaNetForCausalLM', 'PrecondGatedDeltaNetModel']
