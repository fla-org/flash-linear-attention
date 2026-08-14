# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang backend for KDA operations."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend
from fla.utils import check_shared_mem, find_spec_cached, has_usable_nvcc

_TILELANG_AVAILABLE = find_spec_cached("tilelang") is not None


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _tile_extent(dim: int, const_tiling: int) -> int:
    return min(max(_next_power_of_2(dim), 16), const_tiling)


class KDATileLangBackend(BaseBackend):

    backend_type = "tilelang"
    package_name = "tilelang"
    env_var = "FLA_TILELANG"

    @classmethod
    def is_available(cls) -> bool:
        return _TILELANG_AVAILABLE and has_usable_nvcc()

    def chunk_kda_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context=None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if not torch.is_grad_enabled():
            return False, "TileLang KDA public backend currently targets autograd training only"
        if kwargs.keys() - {"chunk_size"}:
            return False, f"TileLang KDA public backend does not support kwargs {sorted(kwargs.keys())}"
        chunk_size = kwargs.get("chunk_size", 64)
        if use_qk_l2norm_in_kernel:
            return False, "TileLang KDA public backend currently supports use_qk_l2norm_in_kernel=False only"
        if use_gate_in_kernel:
            return False, "TileLang KDA public backend currently supports pre-gated KDA only"
        if use_beta_sigmoid_in_kernel:
            return False, "TileLang KDA public backend currently supports post-sigmoid beta only"
        if allow_neg_eigval:
            return False, "TileLang KDA public backend does not support allow_neg_eigval=True"
        if safe_gate:
            return False, "TileLang KDA public backend currently supports safe_gate=False only"
        if disable_recompute:
            return False, "TileLang KDA public backend currently supports disable_recompute=False only"
        if return_intermediate_states:
            return False, "TileLang KDA public backend does not support return_intermediate_states"
        if state_v_first:
            return False, "TileLang KDA public backend currently supports state_v_first=False only"
        if cu_seqlens is not None or cu_seqlens_cpu is not None:
            return False, "TileLang KDA public backend currently supports dense fixed-length sequences only"
        if cp_context is not None:
            return False, "TileLang KDA public backend does not support context parallel"
        if chunk_size != 32:
            return False, f"TileLang KDA public backend supports chunk_size=32 for the measured BF16 bucket, got {chunk_size}"
        tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        if not all(getattr(tensor, "is_cuda", False) for tensor in tensors.values()):
            return False, "TileLang KDA public backend requires CUDA tensors"
        for name, tensor in tensors.items():
            is_contiguous = getattr(tensor, "is_contiguous", None)
            if is_contiguous is not None and not is_contiguous():
                return False, f"TileLang KDA public backend requires {name} to be contiguous"
        if q.dtype != torch.bfloat16:
            return False, f"TileLang KDA public backend requires q dtype torch.bfloat16, got {q.dtype}"
        for name in ("k", "v"):
            if tensors[name].dtype != q.dtype:
                return False, (
                    f"TileLang KDA public backend requires {name} dtype {tensors[name].dtype} "
                    f"to match q dtype {q.dtype}"
                )
        if g.dtype != torch.float32:
            return False, f"TileLang KDA public backend requires g dtype torch.float32, got {g.dtype}"
        if beta.dtype != torch.bfloat16:
            return False, f"TileLang KDA public backend requires beta dtype torch.bfloat16, got {beta.dtype}"
        if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4 or len(g.shape) != 4:
            return False, "TileLang KDA public backend requires q, k, v, and g to be 4D tensors"
        if q.shape != k.shape:
            return False, f"TileLang KDA public backend requires q and k to share shape, got {q.shape} vs {k.shape}"

        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[-1]
        if K != 128 or V != 128:
            return False, f"TileLang KDA public backend supports the measured K=V=128 bucket, got K={K}, V={V}"
        if T % chunk_size != 0:
            return False, (
                f"TileLang KDA public backend requires dense sequence length T={T} to be divisible by "
                f"chunk_size={chunk_size}; fall back to Triton"
            )
        if HV % H != 0:
            return False, f"TileLang KDA public backend requires HV={HV} to be divisible by H={H} for GVA"
        if v.shape != (B, T, HV, V):
            return False, f"TileLang KDA public backend requires v shape {(B, T, HV, V)}, got {v.shape}"
        if g.shape != (B, T, HV, K):
            return False, f"TileLang KDA public backend requires g shape {(B, T, HV, K)}, got {g.shape}"
        if beta.shape != (B, T, HV):
            return False, f"TileLang KDA public backend requires beta shape {(B, T, HV)}, got {beta.shape}"
        if initial_state is not None:
            if initial_state.dtype != torch.float32:
                return False, "TileLang KDA public backend requires initial_state dtype torch.float32"
            is_contiguous = getattr(initial_state, "is_contiguous", None)
            if is_contiguous is not None and not is_contiguous():
                return False, "TileLang KDA public backend requires initial_state to be contiguous"
            if initial_state.shape != (B, HV, K, V):
                return False, (
                    f"TileLang KDA public backend requires initial_state shape {(B, HV, K, V)}, "
                    f"got {initial_state.shape}"
                )
        return True, None

    def chunk_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context=None,
        **kwargs,
    ):
        from fla.ops.kda.chunk import chunk_kda
        return chunk_kda.__wrapped__.__wrapped__(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            _tilelang_helpers=True,
            **kwargs,
        )

    def chunk_kda_bwd_wy_dqkg_fused_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        v_new: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dh: torch.Tensor,
        dv: torch.Tensor,
        scale: float | None = None,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[bool, str | None]:
        if any(getattr(tensor, "requires_grad", False) for tensor in (q, k, v, v_new, g, beta, A, h, do, dh, dv)):
            return False, (
                "TileLang KDA backend is limited to manual fused-backward calls; "
                "use the guarded public chunk_kda backend for autograd TileLang routing"
            )
        data_tensors = {
            "q": q,
            "k": k,
            "v": v,
            "v_new": v_new,
            "A": A,
            "h": h,
            "do": do,
            "dh": dh,
            "dv": dv,
        }
        aux_tensors = {"g": g, "beta": beta}
        if not all(getattr(tensor, "is_cuda", False) for tensor in (*data_tensors.values(), *aux_tensors.values())):
            return False, "TileLang KDA backend requires CUDA tensors"
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, f"TileLang KDA backend does not support dtype {q.dtype}; fall back to Triton"
        if g.dtype != torch.float32:
            return False, f"TileLang KDA backend requires g dtype torch.float32, got {g.dtype}; fall back to Triton"
        if beta.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, f"TileLang KDA backend does not support beta dtype {beta.dtype}; fall back to Triton"
        for name, tensor in data_tensors.items():
            if tensor.dtype != q.dtype:
                return False, f"TileLang KDA backend requires {name} dtype {tensor.dtype} to match q dtype {q.dtype}"
        for name, tensor in {**data_tensors, **aux_tensors}.items():
            is_contiguous = getattr(tensor, "is_contiguous", None)
            if is_contiguous is not None and not is_contiguous():
                return False, f"TileLang KDA backend requires {name} to be contiguous"
        if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4:
            return False, "TileLang KDA backend requires q, k, and v to be 4D tensors"
        if q.shape != k.shape:
            return False, f"TileLang KDA backend requires q and k to have the same shape, got {q.shape} vs {k.shape}"
        if chunk_size not in (32, 64):
            return False, f"TileLang KDA backend supports chunk_size 32 or 64, got {chunk_size}"
        if cu_seqlens is not None or chunk_indices is not None:
            return False, "TileLang KDA backend currently supports dense fixed-length sequences only; fall back to Triton"

        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[-1]
        if T % chunk_size != 0:
            return False, (
                f"TileLang KDA backend requires dense sequence length T={T} to be divisible by "
                f"chunk_size={chunk_size}; fall back to Triton"
            )
        if HV % H != 0:
            return False, (
                f"TileLang KDA backend requires num_v_heads (HV={HV}) to be divisible by "
                f"num_qk_heads (H={H}); HV % H must be 0 for GVA"
            )
        tensor_idx = getattr(getattr(q, "device", None), "index", 0) or 0
        const_tiling = 64 if check_shared_mem(tensor_idx=tensor_idx) else 32
        BK = _tile_extent(K, const_tiling)
        BV = _tile_extent(V, const_tiling)
        if K % BK != 0:
            return False, f"TileLang KDA backend requires K={K} to be divisible by its BK tile {BK}; fall back to Triton"
        if V % BV != 0:
            return False, f"TileLang KDA backend requires V={V} to be divisible by its BV tile {BV}; fall back to Triton"
        if v_new.shape != v.shape or do.shape != v.shape or dv.shape != v.shape:
            return False, (
                "TileLang KDA backend requires v, v_new, do, and dv to share shape "
                f"{v.shape}; got v_new={v_new.shape}, do={do.shape}, dv={dv.shape}"
            )
        if g.shape != (B, T, HV, K):
            return False, f"TileLang KDA backend requires g shape {(B, T, HV, K)}, got {g.shape}"
        if beta.shape != (B, T, HV):
            return False, f"TileLang KDA backend requires beta shape {(B, T, HV)}, got {beta.shape}"
        if A.shape != (B, T, HV, chunk_size):
            return False, f"TileLang KDA backend requires A shape {(B, T, HV, chunk_size)}, got {A.shape}"
        state_shape = (V, K) if state_v_first else (K, V)
        expected_state_shape = (B, T // chunk_size, HV, *state_shape)
        if h.shape != expected_state_shape or dh.shape != expected_state_shape:
            return False, (
                f"TileLang KDA backend requires h/dh shape {expected_state_shape}, "
                f"got h={h.shape}, dh={dh.shape}"
            )
        return True, None

    def chunk_kda_bwd_wy_dqkg_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        v_new: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dh: torch.Tensor,
        dv: torch.Tensor,
        scale: float | None = None,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from fla.ops.kda.backends.tilelang.chunk_bwd_dqkg import (
            chunk_kda_bwd_wy_dqkg_fused_tilelang,
        )
        return chunk_kda_bwd_wy_dqkg_fused_tilelang(
            q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A,
            h=h, do=do, dh=dh, dv=dv,
            scale=scale, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size, chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )
