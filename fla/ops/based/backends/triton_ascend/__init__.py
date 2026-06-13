# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend NPU backend for Based ops."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend


class TritonAscendBasedBackend(BaseBackend):
    """NPU backend for parallel/fused-chunk Based kernels."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def parallel_based_verifier(
        self,
        q,
        k,
        v,
        scale=None,
        use_norm=True,
        head_first=False,
    ):
        if q.shape[-1] > 128:
            return False, "NPU parallel_based supports feature dim up to 128"
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, f"unsupported dtype {q.dtype}"
        return True, None

    def parallel_based(
        self,
        q,
        k,
        v,
        scale=None,
        use_norm=True,
        head_first=False,
    ):
        from fla.ops.based.backends.triton_ascend.parallel import parallel_based_npu
        return parallel_based_npu(q, k, v, scale=scale, use_norm=use_norm, head_first=head_first)

    def fused_chunk_based_verifier(
        self,
        q,
        k,
        v,
        scale=None,
        use_norm=True,
        head_first=False,
    ):
        if q.shape[-1] > 16:
            return False, "NPU fused_chunk_based supports feature dim up to 16"
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, f"unsupported dtype {q.dtype}"
        return True, None

    def fused_chunk_based(
        self,
        q,
        k,
        v,
        scale=None,
        use_norm=True,
        head_first=False,
    ):
        from fla.ops.based.backends.triton_ascend.fused_chunk import fused_chunk_based_npu
        return fused_chunk_based_npu(q, k, v, scale=scale, use_norm=use_norm, head_first=head_first)
