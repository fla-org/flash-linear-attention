# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Module-level backends for FLA components such as rotary and cross-entropy."""

from fla.backends import BackendRegistry

# this import installs the existing NPU GRPO compile workaround before fla.modules.grpo loads.
from fla.modules.backends.triton_ascend import TritonAscendBackend

modules_registry = BackendRegistry("modules")

modules_registry.register(TritonAscendBackend())

dispatch = modules_registry.dispatch

__all__ = ['dispatch', 'modules_registry']
