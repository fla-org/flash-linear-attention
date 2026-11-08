# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""SwiLA hybrid configuration names backed by FLA's shared validation."""

from fla.models.configuration_utils import AttentionConfig as HybridAttentionConfig
from fla.models.configuration_utils import AttentionSpec as HybridAttentionSpec
from fla.models.configuration_utils import parse_attention_config as normalize_hybrid_attention_config

__all__ = ['HybridAttentionConfig', 'HybridAttentionSpec', 'normalize_hybrid_attention_config']
