# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.swila.configuration_swila import SwiLAConfig
from fla.models.swila.modeling_swila import SwiLABlock, SwiLAForCausalLM, SwiLAModel, SwiLAPreTrainedModel


class _LegacyModelTypeConfig(SwiLAConfig):
    """Read the historical model type into the canonical configuration class."""

    model_type = "sla"

    def __new__(cls, *args, **kwargs):
        # AutoConfig.for_model constructs the registered class directly.
        # Return a canonical instance here as well, without exposing a second
        # model family or serializing the historical registration marker.
        return SwiLAConfig(*args, **kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # AutoConfig requires the registration key to match this class marker,
        # but its result must use the canonical class for AutoModel dispatch.
        return SwiLAConfig.from_dict(config_dict, **kwargs)


AutoConfig.register("swila", SwiLAConfig, exist_ok=True)
AutoConfig.register("sla", _LegacyModelTypeConfig, exist_ok=True)
AutoModel.register(SwiLAConfig, SwiLAModel, exist_ok=True)
AutoModelForCausalLM.register(SwiLAConfig, SwiLAForCausalLM, exist_ok=True)

__all__ = ['SwiLABlock', 'SwiLAConfig', 'SwiLAForCausalLM', 'SwiLAModel', 'SwiLAPreTrainedModel']
