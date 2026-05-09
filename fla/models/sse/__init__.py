# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.sse.configuration_sse import SSEConfig
from fla.models.sse.modeling_sse import SSEForCausalLM, SSEModel

AutoConfig.register(SSEConfig.model_type, SSEConfig, exist_ok=True)
AutoModel.register(SSEConfig, SSEModel, exist_ok=True)
AutoModelForCausalLM.register(SSEConfig, SSEForCausalLM, exist_ok=True)

__all__ = ['SSEConfig', 'SSEForCausalLM', 'SSEModel']
