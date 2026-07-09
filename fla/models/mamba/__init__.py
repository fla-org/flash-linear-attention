# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mamba.configuration_mamba import MambaConfig
from fla.models.mamba.modeling_mamba import MambaForCausalLM, MambaModel

AutoConfig.register(MambaConfig.model_type, MambaConfig, exist_ok=True)
AutoModel.register(MambaConfig, MambaModel, exist_ok=True)
AutoModelForCausalLM.register(MambaConfig, MambaForCausalLM, exist_ok=True)


__all__ = ['MambaConfig', 'MambaForCausalLM', 'MambaModel']
