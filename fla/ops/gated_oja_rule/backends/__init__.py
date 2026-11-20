# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# For a list of contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Gated Oja rule backends."""

from fla.ops.gated_oja_rule.backends.triton_ascend import TritonAscendGatedOjaRuleBackend

__all__ = ['TritonAscendGatedOjaRuleBackend']
