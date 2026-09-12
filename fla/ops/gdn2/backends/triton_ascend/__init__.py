# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend backend for GDN-2."""

from __future__ import annotations

import torch
import triton

from fla.ops.backends import BaseBackend

# The grouped diagonal kernel keeps the padded K dimension in one UB slab.
_MAX_FWD_INTRA_BK = 256


class TritonAscendGDN2Backend(BaseBackend):
    """Ascend NPU backend for GDN-2 chunk kernels."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_gdn2_fwd_intra_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        b: torch.Tensor,
        w_gate: torch.Tensor,
        scale: float,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
        safe_gate: bool = False,
        disable_recompute: bool = False,
    ) -> tuple[bool, str | None]:
        del scale, safe_gate, disable_recompute
        if chunk_size != 64:
            return False, f"GDN-2 Ascend intra requires chunk_size=64, got {chunk_size}"
        K = int(k.shape[-1])
        BK = triton.next_power_of_2(K)
        if BK > _MAX_FWD_INTRA_BK:
            return False, (
                f"GDN-2 Ascend intra requires next_power_of_2(K) <= {_MAX_FWD_INTRA_BK} "
                f"for UB capacity, got K={K} (BK={BK})"
            )
        float_tensors = (q, k, v, gk, b, w_gate)
        tensors = (*float_tensors, *(t for t in (cu_seqlens, chunk_indices) if t is not None))
        if any(t.device.type != "npu" for t in tensors):
            return False, "GDN-2 Ascend intra requires NPU tensors"
        supported = (torch.float16, torch.bfloat16, torch.float32)
        if any(t.dtype not in supported for t in float_tensors):
            return False, "GDN-2 Ascend intra received an unsupported dtype"
        return True, None

    def chunk_gdn2_fwd_intra(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.chunk_intra import chunk_gdn2_fwd_intra_npu
        return chunk_gdn2_fwd_intra_npu(*args, **kwargs)

    def recompute_w_u_fwd_gdn2_verifier(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        w_gate: torch.Tensor,
        A: torch.Tensor,
        q: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[bool, str | None]:
        if gk is None:
            return False, "GDN-2 Ascend WY requires gk"
        float_tensors = (k, v, b, w_gate, A, gk, *(t for t in (q,) if t is not None))
        if any(t.ndim != 4 for t in float_tensors):
            return False, "GDN-2 Ascend WY requires rank-4 tensors"
        if A.shape[-1] != 64:
            return False, f"GDN-2 Ascend WY requires chunk_size=64, got {A.shape[-1]}"
        if A.shape[:3] != k.shape[:3]:
            return False, "GDN-2 Ascend WY requires A to match k on [B, T, H]"
        if any(t.shape != k.shape for t in (b, gk, *(t for t in (q,) if t is not None))):
            return False, "GDN-2 Ascend WY requires b/gk/q shapes to match k"
        if v.shape[:3] != k.shape[:3] or w_gate.shape != v.shape:
            return False, "GDN-2 Ascend WY requires v/w_gate to match k on [B, T, H]"
        if cu_seqlens is not None and k.shape[0] != 1:
            return False, "GDN-2 Ascend WY requires batch size 1 for packed sequences"
        if cu_seqlens is not None and cu_seqlens.ndim != 1:
            return False, "GDN-2 Ascend WY requires cu_seqlens to be rank 1"
        if chunk_indices is not None and (chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2):
            return False, "GDN-2 Ascend WY requires chunk_indices shape [NT, 2]"
        K = int(k.shape[-1])
        if not 1 <= K <= 256:
            return False, f"GDN-2 Ascend WY requires 1 <= K <= 256, got K={K}"
        V = int(v.shape[-1])
        if V < 1:
            return False, f"GDN-2 Ascend WY requires V >= 1, got V={V}"
        tensors = (*float_tensors, *(t for t in (cu_seqlens, chunk_indices) if t is not None))
        if any(t.device.type != "npu" or t.device != k.device for t in tensors):
            return False, "GDN-2 Ascend WY requires tensors on the same NPU device"
        supported = (torch.float16, torch.bfloat16, torch.float32)
        if any(t.dtype not in supported for t in float_tensors):
            return False, "GDN-2 Ascend WY received an unsupported dtype"
        if A.dtype != k.dtype or A.dtype != v.dtype:
            return False, "GDN-2 Ascend WY requires A, k, and v to have the same dtype"
        index_tensors = tuple(t for t in (cu_seqlens, chunk_indices) if t is not None)
        if any(t.dtype not in (torch.int32, torch.int64) for t in index_tensors):
            return False, "GDN-2 Ascend WY requires int32 or int64 sequence indices"
        return True, None

    def recompute_w_u_fwd_gdn2(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.wy_fast import recompute_w_u_fwd_gdn2_npu
        return recompute_w_u_fwd_gdn2_npu(*args, **kwargs)

    def chunk_gdn2_bwd_wy_dqkg_fused_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        v_new: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w_gate: torch.Tensor,
        A: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dh: torch.Tensor,
        dv: torch.Tensor,
        scale: float | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
        state_v_first: bool = False,
    ) -> tuple[bool, str | None]:
        del scale, state_v_first
        if chunk_size != 64:
            return False, f"GDN-2 Ascend backward requires chunk_size=64, got {chunk_size}"
        float_tensors = (q, k, v, v_new, g, b, w_gate, A, h, do, dh, dv)
        tensors = (*float_tensors, *(t for t in (cu_seqlens, chunk_indices) if t is not None))
        if any(t.device.type != "npu" for t in tensors):
            return False, "GDN-2 Ascend backward requires NPU tensors"
        supported = (torch.float16, torch.bfloat16, torch.float32)
        if any(t.dtype not in supported for t in float_tensors):
            return False, "GDN-2 Ascend backward received an unsupported dtype"
        return True, None

    def chunk_gdn2_bwd_wy_dqkg_fused(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.chunk_bwd import chunk_gdn2_bwd_wy_dqkg_fused_npu
        return chunk_gdn2_bwd_wy_dqkg_fused_npu(*args, **kwargs)
