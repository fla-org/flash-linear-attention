# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Wall attention, contributed by Tilde Research (Timor Averbuch, Dhruv Pai).

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.wall_transformer.configuration_wall_transformer import WallTransformerConfig
from fla.models.wall_transformer.modeling_wall_transformer import (
    WallTransformerForCausalLM,
    WallTransformerModel,
)

AutoConfig.register(WallTransformerConfig.model_type, WallTransformerConfig, exist_ok=True)
AutoModel.register(WallTransformerConfig, WallTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(WallTransformerConfig, WallTransformerForCausalLM, exist_ok=True)


__all__ = ['WallTransformerConfig', 'WallTransformerForCausalLM', 'WallTransformerModel']
