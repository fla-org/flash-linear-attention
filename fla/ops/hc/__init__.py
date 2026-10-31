# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import GluonBackend, TileLangBackend
from fla.ops.hc.fused import fused_hc_post, fused_hc_pre
from fla.ops.hc.naive import naive_hc_post, naive_hc_pre

__all__ = [
    'fused_hc_post',
    'fused_hc_pre',
    'naive_hc_post',
    'naive_hc_pre',
]

if GluonBackend.is_available():
    from fla.ops.hc import gluon  # noqa: F401

if TileLangBackend.is_available():
    from fla.ops.hc import tilelang  # noqa: F401
