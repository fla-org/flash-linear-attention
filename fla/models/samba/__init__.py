# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.samba.configuration_samba import SambaConfig
from fla.models.samba.modeling_samba import SambaForCausalLM, SambaModel

AutoConfig.register(SambaConfig.model_type, SambaConfig, exist_ok=True)
AutoModel.register(SambaConfig, SambaModel, exist_ok=True)
AutoModelForCausalLM.register(SambaConfig, SambaForCausalLM, exist_ok=True)


__all__ = ['SambaConfig', 'SambaForCausalLM', 'SambaModel']
