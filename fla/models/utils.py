# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compatibility imports for model utilities."""

from fla.models.cache_utils import Cache, FLACache, FLALayer, LegacyFLACache
from fla.models.generation_utils import FLAGenerationMixin, FLAUnsupportedCacheGenerationMixin

__all__ = ['Cache', 'FLACache', 'FLAGenerationMixin', 'FLALayer', 'FLAUnsupportedCacheGenerationMixin', 'LegacyFLACache']
