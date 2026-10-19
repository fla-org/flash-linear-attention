# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Common backends for shared operations like chunk_gated_delta_rule_fwd_h."""

from fla.backends import TritonAscendBackend
from fla.ops.common.backends.intracard import IntraCardCPBackend  # noqa: F401
from fla.ops.common.backends.tilelang import CommonTileLangBackend  # noqa: F401

if TritonAscendBackend.is_available():
    from fla.ops.common.backends.triton_ascend import chunk_delta_h, chunk_h, chunk_o, chunk_scaled_dot_kkt, gate  # noqa: F401
