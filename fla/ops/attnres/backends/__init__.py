# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""AttnRes backends."""

from fla.backends import GluonBackend, TritonAscendBackend

if GluonBackend.is_available():
    from fla.ops.attnres.backends.gluon import AttnResGluonBackend  # noqa: F401

if TritonAscendBackend.is_available():
    from fla.ops.attnres.backends.triton_ascend import fused  # noqa: F401
