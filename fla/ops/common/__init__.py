# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TileLangBackend, TritonAscendBackend
from fla.ops.common.backends import intracard  # noqa: F401

if TritonAscendBackend.is_available():
    from fla.ops.common.backends.triton_ascend import chunk_delta_h, chunk_h, chunk_o, chunk_scaled_dot_kkt, gate  # noqa: F401

if TileLangBackend.is_available():
    from fla.ops.common.backends import tilelang  # noqa: F401
