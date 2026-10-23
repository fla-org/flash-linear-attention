# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CP backends. Importing this package registers them with ``@dispatch('cp')``."""

from fla.ops.cp.backends.triton_ascend import TritonAscendCPBackend

__all__ = ['TritonAscendCPBackend']


def _bind_npu_kernels() -> None:
    """Point direct kernel imports at the Ascend implementations.

    ``intracard`` launches ``pre_process_fwd_kernel_merged`` and
    ``merge_fwd_bwd_kernel`` by name. Those symbols are not dispatched
    entry points, so the NPU kernels have to replace them on the module.
    """
    from fla.utils import IS_NPU
    if not IS_NPU:
        return
    import fla.ops.cp.chunk_delta_h as cp_chunk_delta_h
    from fla.ops.cp.backends.triton_ascend.chunk_delta_h import (
        merge_fwd_bwd_kernel_npu,
        pre_process_bwd_kernel_merged_npu,
        pre_process_fwd_kernel_merged_npu,
    )
    cp_chunk_delta_h.pre_process_fwd_kernel_merged = pre_process_fwd_kernel_merged_npu
    cp_chunk_delta_h.pre_process_bwd_kernel_merged = pre_process_bwd_kernel_merged_npu
    cp_chunk_delta_h.merge_fwd_bwd_kernel = merge_fwd_bwd_kernel_npu


_bind_npu_kernels()
