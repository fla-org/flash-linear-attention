# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import GluonBackend, TileLangBackend
from fla.ops.utils.sinkhorn.naive import naive_sinkhorn
from fla.ops.utils.sinkhorn.ops import sinkhorn

__all__ = ['naive_sinkhorn', 'sinkhorn']


if GluonBackend.is_available():
    from fla.ops.utils.sinkhorn import gluon  # noqa: F401

if TileLangBackend.is_available():
    from fla.ops.utils.sinkhorn import tilelang  # noqa: F401
