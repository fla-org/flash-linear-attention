# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend backend for GDN-2."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend


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
        float_tensors = (q, k, v, gk, b, w_gate)
        tensors = (*float_tensors, *(t for t in (cu_seqlens, chunk_indices) if t is not None))
        if any(t.device.type != "npu" for t in tensors):
            return False, "GDN-2 Ascend intra requires NPU tensors"
        supported = (torch.float16, torch.bfloat16, torch.float32)
        if any(t.dtype not in supported for t in float_tensors):
            return False, "GDN-2 Ascend intra received an unsupported dtype"
        return True, None

    def chunk_gdn2_fwd_intra(
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
    ):
        from fla.ops.gdn2.backends.triton_ascend.chunk_intra import chunk_gdn2_fwd_intra_npu
        return chunk_gdn2_fwd_intra_npu(
            q,
            k,
            v,
            gk,
            b,
            w_gate,
            scale,
            cu_seqlens,
            chunk_size,
            chunk_indices,
            safe_gate,
            disable_recompute,
        )

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

    def chunk_gdn2_bwd_wy_dqkg_fused(
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
    ):
        from fla.ops.gdn2.backends.triton_ascend.chunk_bwd import chunk_gdn2_bwd_wy_dqkg_fused_npu
        return chunk_gdn2_bwd_wy_dqkg_fused_npu(
            q,
            k,
            v,
            v_new,
            g,
            b,
            w_gate,
            A,
            h,
            do,
            dh,
            dv,
            scale,
            cu_seqlens,
            chunk_size,
            chunk_indices,
            state_v_first,
        )
