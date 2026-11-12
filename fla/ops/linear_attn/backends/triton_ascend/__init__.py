# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend backend for un-gated linear attention."""

from __future__ import annotations

import torch

from fla.backends import TritonAscendBackend, register

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _get_tensor(args: tuple, kwargs: dict, name: str, index: int):
    return kwargs.get(name, args[index] if len(args) > index else None)


@register('linear_attn')
class TritonAscendLinearAttentionBackend(TritonAscendBackend):
    """Ascend NPU implementations for the linear-attention public entries."""

    def _verify_common(self, *args, **kwargs) -> tuple[bool, str | None]:
        q = _get_tensor(args, kwargs, 'q', 0)
        k = _get_tensor(args, kwargs, 'k', 1)
        v = _get_tensor(args, kwargs, 'v', 2)
        tensors = (q, k, v)
        if any(tensor is None for tensor in tensors):
            return False, 'q, k, and v are required'
        if any(tensor.device.type != 'npu' for tensor in tensors):
            return False, 'q, k, and v must be on NPU'
        if any(tensor.dtype not in _SUPPORTED_DTYPES for tensor in tensors):
            return False, 'only fp16, bf16, and fp32 are supported'
        if any(tensor.ndim != 4 for tensor in tensors):
            return False, 'q, k, and v must be rank-4 tensors'
        cu_seqlens = kwargs.get('cu_seqlens')
        if cu_seqlens is not None and q.shape[0] != 1:
            return False, 'packed variable-length input requires B=1'
        return True, None

    def chunk_linear_attn_verifier(self, *args, **kwargs):
        return self._verify_common(*args, **kwargs)

    def naive_chunk_linear_attn_verifier(self, *args, **kwargs):
        return self._verify_common(*args, **kwargs)

    def naive_chunk_linear_attn(self, *args, **kwargs):
        from fla.ops.linear_attn.backends.triton_ascend.naive_chunk import (
            naive_chunk_linear_attn as naive_chunk_linear_attn_npu,
        )
        return naive_chunk_linear_attn_npu(*args, **kwargs)

    def chunk_linear_attn(self, *args, **kwargs):
        from fla.ops.linear_attn.backends.triton_ascend.linear_attn import (
            chunk_linear_attn as chunk_linear_attn_npu,
        )
        return chunk_linear_attn_npu(*args, **kwargs)

    def fused_chunk_linear_attn_verifier(self, *args, **kwargs):
        return self._verify_common(*args, **kwargs)

    def fused_chunk_linear_attn(self, *args, **kwargs):
        from fla.ops.linear_attn.backends.triton_ascend.fused_linear_attn import (
            fused_chunk_linear_attn as fused_chunk_linear_attn_npu,
        )
        return fused_chunk_linear_attn_npu(*args, **kwargs)

    def fused_recurrent_linear_attn_verifier(self, *args, **kwargs):
        return self._verify_common(*args, **kwargs)

    def fused_recurrent_linear_attn(self, *args, **kwargs):
        from fla.ops.linear_attn.backends.triton_ascend.fused_linear_attn import (
            fused_recurrent_linear_attn as fused_recurrent_linear_attn_npu,
        )
        return fused_recurrent_linear_attn_npu(*args, **kwargs)
