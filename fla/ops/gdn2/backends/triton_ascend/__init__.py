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

from fla.backends import BaseBackend, register_backend

# keep the backend verifier aligned with chunk_gdn2's public K <= 256 contract.
_MAX_FWD_INTRA_BK = 256


@register_backend('gdn2')
class TritonAscendGDN2Backend(BaseBackend):
    """Ascend NPU backend for GDN-2 kernels."""

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

    def chunk_gdn2_bwd_intra_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        dAqk: torch.Tensor,
        dAkk: torch.Tensor,
        dq: torch.Tensor,
        dk: torch.Tensor,
        db: torch.Tensor,
        dg: torch.Tensor,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        chunk_size: int = 64,
        safe_gate: bool = False,
    ) -> tuple[bool, str | None]:
        del safe_gate
        if chunk_size != 64:
            return False, f"GDN-2 Ascend bwd intra requires chunk_size=64, got {chunk_size}"
        float_tensors = (q, k, g, b, dAqk, dAkk, dq, dk, db, dg)
        if any(t.ndim != 4 for t in float_tensors):
            return False, "GDN-2 Ascend bwd intra requires rank-4 tensors"
        if cu_seqlens is not None and k.shape[0] != 1:
            return False, "GDN-2 Ascend bwd intra requires batch size 1 for packed sequences"
        if cu_seqlens is not None and cu_seqlens.ndim != 1:
            return False, "GDN-2 Ascend bwd intra requires rank-1 cu_seqlens"
        if chunk_indices is not None and (chunk_indices.ndim != 2 or chunk_indices.shape[-1] != 2):
            return False, "GDN-2 Ascend bwd intra requires chunk_indices shaped [NT, 2]"
        if any(t.shape != k.shape for t in (q, g, b, dq, dk, db, dg)):
            return False, "GDN-2 Ascend bwd intra requires q/k/g/b and gradient shapes to match"
        if dAqk.shape != dAkk.shape or dAqk.shape[:3] != k.shape[:3] or dAqk.shape[-1] != chunk_size:
            return False, "GDN-2 Ascend bwd intra requires dAqk/dAkk shaped [B, T, H, 64]"
        K = int(k.shape[-1])
        if not 1 <= K <= 256:
            return False, f"GDN-2 Ascend bwd intra requires 1 <= K <= 256, got K={K}"
        tensors = (*float_tensors, *(t for t in (cu_seqlens, chunk_indices) if t is not None))
        if any(t.device.type != "npu" or t.device != k.device for t in tensors):
            return False, "GDN-2 Ascend bwd intra requires tensors on the same NPU device"
        supported = (torch.float16, torch.bfloat16, torch.float32)
        if any(t.dtype not in supported for t in float_tensors):
            return False, "GDN-2 Ascend bwd intra received an unsupported dtype"
        index_tensors = tuple(t for t in (cu_seqlens, chunk_indices) if t is not None)
        if any(t.dtype not in (torch.int32, torch.int64) for t in index_tensors):
            return False, "GDN-2 Ascend bwd intra requires int32 or int64 sequence indices"
        return True, None

    def chunk_gdn2_bwd_intra(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.chunk_intra import chunk_gdn2_bwd_intra_npu
        return chunk_gdn2_bwd_intra_npu(*args, **kwargs)

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

    def fused_recurrent_gdn2_fwd_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        scale: float | None = None,
        output_final_state: bool = False,
        inplace_final_state: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        lower_bound: float | None = None,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        del scale, output_final_state, inplace_final_state, state_v_first
        del use_qk_l2norm_in_kernel, use_gate_in_kernel, lower_bound, kwargs
        required = {'q': q, 'k': k, 'v': v, 'g': g, 'b': b, 'w': w}
        if any(not isinstance(t, torch.Tensor) for t in required.values()):
            return False, 'GDN-2 Ascend recurrent requires tensor inputs'
        optional = (A_log, dt_bias, initial_state, cu_seqlens, ssm_state_indices, num_accepted_tokens, out)
        bad_device = [name for name, t in required.items() if t.device.type != 'npu']
        bad_device.extend(
            f'optional[{i}]' for i, t in enumerate(optional)
            if isinstance(t, torch.Tensor) and t.device.type != 'npu'
        )
        if bad_device:
            return False, f'GDN-2 Ascend recurrent requires NPU tensors, got {bad_device}'
        float_tensors = (
            *required.values(),
            *(t for t in (A_log, dt_bias, initial_state, out) if isinstance(t, torch.Tensor)),
        )
        supported = (torch.float16, torch.bfloat16, torch.float32)
        if any(t.dtype not in supported for t in float_tensors):
            return False, 'GDN-2 Ascend recurrent received an unsupported dtype'
        return True, None

    def fused_recurrent_gdn2_fwd(self, *args, **kwargs):
        from fla.ops.gdn2.backends.triton_ascend.fused_recurrent import fused_recurrent_gdn2_fwd_npu
        return fused_recurrent_gdn2_fwd_npu(*args, **kwargs)
