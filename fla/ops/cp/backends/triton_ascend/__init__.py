# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend backend for CP pre-process and merge."""

from __future__ import annotations

from fla.backends import BaseBackend, register


def _verify_npu_input(args, kwargs) -> tuple[bool, str | None]:
    tensor = args[0] if args else kwargs.get('k')
    if tensor is None or tensor.device.type != 'npu':
        return False, 'input device is not NPU'
    return True, None


@register('cp')
class TritonAscendCPBackend(BaseBackend):
    """Ascend implementation of the CP state pre-process and merge entry points."""

    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_gated_delta_rule_fwd_h_pre_process_verifier(self, *args, **kwargs):
        return _verify_npu_input(args, kwargs)

    def chunk_gated_delta_rule_fwd_h_pre_process(self, *args, **kwargs):
        from fla.ops.cp.backends.triton_ascend.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h_pre_process_npu,
        )
        return chunk_gated_delta_rule_fwd_h_pre_process_npu(*args, **kwargs)

    def chunk_gated_delta_rule_bwd_dhu_pre_process_verifier(self, *args, **kwargs):
        return _verify_npu_input(args, kwargs)

    def chunk_gated_delta_rule_bwd_dhu_pre_process(self, *args, **kwargs):
        from fla.ops.cp.backends.triton_ascend.chunk_delta_h import (
            chunk_gated_delta_rule_bwd_dhu_pre_process_npu,
        )
        return chunk_gated_delta_rule_bwd_dhu_pre_process_npu(*args, **kwargs)
