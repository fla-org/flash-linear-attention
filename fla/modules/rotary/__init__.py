# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TritonAscendBackend
from fla.modules.rotary.ops import RotaryEmbedding, rotary_embedding

__all__ = ['RotaryEmbedding', 'rotary_embedding']


if TritonAscendBackend.is_available():
    from fla.modules.rotary import triton_ascend  # noqa: F401
