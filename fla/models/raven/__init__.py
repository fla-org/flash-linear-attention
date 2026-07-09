# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.raven.configuration_raven import RavenConfig
from fla.models.raven.modeling_raven import RavenForCausalLM, RavenModel

AutoConfig.register(RavenConfig.model_type, RavenConfig, exist_ok=True)
AutoModel.register(RavenConfig, RavenModel, exist_ok=True)
AutoModelForCausalLM.register(RavenConfig, RavenForCausalLM, exist_ok=True)


__all__ = ['RavenConfig', 'RavenForCausalLM', 'RavenModel']
