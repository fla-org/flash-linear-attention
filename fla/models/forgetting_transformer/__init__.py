# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.forgetting_transformer.configuration_forgetting_transformer import ForgettingTransformerConfig
from fla.models.forgetting_transformer.modeling_forgetting_transformer import (
    ForgettingTransformerForCausalLM,
    ForgettingTransformerModel,
)

AutoConfig.register(ForgettingTransformerConfig.model_type, ForgettingTransformerConfig, exist_ok=True)
AutoModel.register(ForgettingTransformerConfig, ForgettingTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(ForgettingTransformerConfig, ForgettingTransformerForCausalLM, exist_ok=True)


__all__ = ['ForgettingTransformerConfig', 'ForgettingTransformerForCausalLM', 'ForgettingTransformerModel']
