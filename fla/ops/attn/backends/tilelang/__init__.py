# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from fla.backends import TileLangBackend
from fla.utils import IS_NVIDIA_HOPPER, TRITON_ABOVE_3_4_0


class AttnTileLangBackend(TileLangBackend):
    # work around Hopper regressions with Triton 3.4+ (see #640).
    default_enable = IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0
