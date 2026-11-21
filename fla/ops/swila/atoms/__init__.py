# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Shared Triton primitives, including active reduction workarounds."""

from fla.ops.swila.atoms.activations import logaddexp, logsigmoid_pair
from fla.ops.swila.atoms.reductions import dot_3d_axis2, sum_2d_axis1
from fla.ops.swila.atoms.softmax import log_softmax_2d, softmax_2d, softmax_backward_2d
from fla.ops.swila.atoms.state_ops import prediction, readout, state_update
