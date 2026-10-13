# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""AttnRes backends."""

from fla.ops.attnres.backends.triton_ascend import TritonAscendAttnResBackend

__all__ = ['TritonAscendAttnResBackend']

# gluon.py imports triton.experimental at module load; keep the Ascend backend usable when Gluon is unavailable.
try:
    from fla.ops.attnres.backends.gluon import AttnResGluonBackend

    __all__.append('AttnResGluonBackend')
except ImportError:
    pass
