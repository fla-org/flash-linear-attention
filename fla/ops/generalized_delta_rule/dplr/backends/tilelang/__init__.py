# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang backend for DPLR operations."""

from __future__ import annotations

import logging

import torch

from fla.ops.backends import BaseBackend
from fla.utils import (
    find_spec_cached,
    get_device_capability,
    get_device_smem_optin,
    get_multiprocessor_count,
    has_usable_nvcc,
)

from .schedules import chunk64_schedule_or_none, stream_bwd_schedule_or_none

logger = logging.getLogger(__name__)

_TILELANG_AVAILABLE = find_spec_cached("tilelang") is not None

_FALLBACK_LOGGED: set[str] = set()


class DPLRTileLangBackend(BaseBackend):

    backend_type = "tilelang"
    package_name = "tilelang"
    env_var = "FLA_TILELANG"

    @classmethod
    def is_available(cls) -> bool:
        return _TILELANG_AVAILABLE and has_usable_nvcc()

    def chunk_dplr_delta_rule_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        chunk_size: int | None = None,
        disable_recompute: bool = False,
        cp_context=None,
        lower_bound: float | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if lower_bound is not None and lower_bound >= 0:
            # rejected so the default implementation surfaces the ValueError
            return False, "TileLang backend requires `lower_bound < 0`; fall back to Triton"
        if cp_context is not None:
            if initial_state is not None:
                return False, "TileLang backend does not support initial_state with CP; fall back to Triton"
            if output_final_state:
                return False, "TileLang backend does not support output_final_state with CP; fall back to Triton"
            if getattr(cp_context, "cu_seqlens", None) is None:
                return False, "TileLang backend requires cu_seqlens for CP; fall back to Triton"
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False, f"TileLang backend does not support dtype {q.dtype}; fall back to Triton"
        if not all(t.dtype == q.dtype for t in (k, v, a, b)):
            return False, (
                "TileLang backend requires k/v/a/b dtypes to match q.dtype "
                f"(got q={q.dtype}, k={k.dtype}, v={v.dtype}, a={a.dtype}, b={b.dtype}); "
                "fall back to Triton"
            )
        if gk.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, f"TileLang backend does not support gk dtype {gk.dtype}; fall back to Triton"
        if k.shape[-1] != v.shape[-1]:
            return False, (
                f"TileLang backend requires K == V (got K={k.shape[-1]}, V={v.shape[-1]}); "
                "fall back to Triton"
            )
        if k.shape[-1] not in (64, 128):
            return False, f"TileLang backend supports head dim 64 or 128 (got {k.shape[-1]}); fall back to Triton"
        from fla.ops.generalized_delta_rule.dplr.chunk import gate_bound_is_safe
        if chunk_size is None:
            chunk_size = 64 if lower_bound is not None and gate_bound_is_safe(lower_bound, 64) else 16
        if chunk_size not in (16, 32, 64):
            return False, f"TileLang backend supports chunk_size 16/32/64, got {chunk_size}; fall back to Triton"
        bound = lower_bound if lower_bound is not None else (-5.0 if safe_gate else None)
        if bound is None or not gate_bound_is_safe(bound, chunk_size):
            return False, (
                "TileLang backend requires safe_gate or a lower_bound that fits "
                f"chunk_size {chunk_size}: its mid-chunk-centered tensor-core scheme "
                "keeps exp2 operands in fp32 range only while "
                "(chunk_size/2+1)*bound*log2(e) <= 124 (the documented safe_gate "
                "range [-5, 0) fits chunk_size 16/32 but not 64); fall back to Triton"
            )
        if not q.is_cuda:
            return False, "TileLang backend is CUDA-only; fall back to Triton"
        dev = q.device.index or 0
        cc_major, cc_minor = get_device_capability(dev)
        smem_cap = get_device_smem_optin(dev)
        K = k.shape[-1]
        in_dtype = "float16" if q.dtype == torch.float16 else "bfloat16"
        if chunk_size == 16 and K == 128:
            # measured ~0.5x vs Triton (the non-vectorized A-stage and the
            # 2-warp h+o path at BT=16 do not pay off at K=128)
            return False, "TileLang backend is slower than Triton at chunk_size 16 with head dim 128; fall back to Triton"
        if chunk_size == 16 and (cu_seqlens is not None or cp_context is not None) and cc_major < 12:
            # measured 0.83-1.02x vs Triton on sm_90 (H800, D64, all varlen
            # sizes): the BT=16 serial state pass does not amortize ragged
            # chunks there. Rect cs16 and cc12x varlen cs16 win and stay
            # accepted.
            return False, (
                "TileLang backend is slower than Triton at chunk_size 16 on "
                f"variable-length inputs on compute capability {cc_major}.{cc_minor}; "
                "fall back to Triton"
            )
        if chunk_size == 64:
            # reject configs no BT=64 kernel schedule can launch on this
            # device (e.g. the K=128 stream backward needs 167936B on A100's
            # 166912B cap or cc120's 101376B cap, where K=64 fits via the
            # low schedule);
            # the arithmetic is shared with the launcher so acceptance
            # implies schedulability
            if chunk64_schedule_or_none(K=K, V=K, in_dtype=in_dtype, smem_cap=smem_cap,
                                        cc=cc_major * 10 + cc_minor) is None:
                return False, (
                    f"TileLang backend has no launchable backward schedule for "
                    f"chunk_size 64 with head dim {K} on a device with {smem_cap}B "
                    "shared memory per block; fall back to Triton"
                )
            if cc_major < 12:
                # measured 0.68-1.05x vs Triton on sm_90 (H800, rect and
                # varlen, both head dims): the BT=64 wy/wu UT-transform
                # stages are latency-bound there, and the fused h+o pass
                # does not amortize at twice the chunk serial length
                return False, (
                    "TileLang backend is slower than Triton at chunk_size 64 "
                    f"on compute capability {cc_major}.{cc_minor}; fall back to Triton"
                )
        # The fused h+o state pass parallelizes only over (V/BV, N, H) blocks
        # and walks chunks serially, so it loses to the split Triton kernels
        # when the grid underfills the GPU. Measured crossover on PRO 6000 /
        # H100 class parts is around half the SM count.
        bv = 64 if v.shape[-1] <= 64 else 32
        if cp_context is not None:
            n_seqs = len(cp_context.cu_seqlens) - 1
        elif cu_seqlens is not None:
            n_seqs = len(cu_seqlens) - 1
        else:
            n_seqs = q.shape[0]
        grid = n_seqs * q.shape[2] * ((v.shape[-1] + bv - 1) // bv)
        sm = get_multiprocessor_count(dev)
        if grid < sm // 2:
            return False, (
                f"TileLang backend is slower than Triton on small grids (N*H*(V/BV)={grid} "
                f"< {sm // 2} SMs/2); fall back to Triton"
            )
        stream_schedule = stream_bwd_schedule_or_none(
            K=K, V=K, BT=chunk_size, in_dtype=in_dtype,
            smem_cap=smem_cap,
        )
        if stream_schedule is None:
            return False, (
                f"TileLang backend has no launchable backward schedule for "
                f"chunk_size {chunk_size} with head dim {K} on a device with {smem_cap}B "
                "shared memory per block; fall back to Triton"
            )
        if stream_schedule == "low" and n_seqs * q.shape[2] < sm // 2:
            # the low-smem stream backward is one serial chunk scan per
            # (seq, head) block with no V split, and its 97KB footprint leaves
            # no room to prefetch; below half the SMs it cannot hide the
            # serial chain and measurably loses to the split Triton kernels
            return False, (
                f"TileLang backend is slower than Triton when the low-smem stream backward "
                f"underfills the device (N*H={n_seqs * q.shape[2]} < {sm // 2} SMs/2); "
                "fall back to Triton"
            )
        return True, None

    def chunk_dplr_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        chunk_size: int | None = None,
        disable_recompute: bool = False,
        cp_context=None,
        lower_bound: float | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.ops.generalized_delta_rule.dplr.backends.tilelang.chunk import (
            chunk_dplr_delta_rule_tilelang,
        )
        try:
            return chunk_dplr_delta_rule_tilelang(
                q=q, k=k, v=v, a=a, b=b, gk=gk,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                chunk_size=chunk_size,
                disable_recompute=disable_recompute,
                cp_context=cp_context,
            )
        except Exception as exc:
            # The verifier gates on the schedule arithmetic, but JIT/launch
            # failures can still escape; honor the dispatch contract and fall
            # back to the default Triton implementation. Only the forward call
            # is guarded: once it succeeds, autograd is committed to the
            # TileLang backward.
            if cp_context is not None:
                # a rank-local fallback would diverge the CP collectives
                raise
            key = f"{type(exc).__name__}: {exc}"
            if key not in _FALLBACK_LOGGED:
                _FALLBACK_LOGGED.add(key)
                logger.warning(
                    f"[FLA Backend] TileLang DPLR forward failed ({key}); falling back to Triton"
                )
            from fla.ops.generalized_delta_rule.dplr import chunk as dplr_chunk
            fn = dplr_chunk.chunk_dplr_delta_rule
            while hasattr(fn, "__wrapped__"):
                fn = fn.__wrapped__
            return fn(
                q=q, k=k, v=v, a=a, b=b, gk=gk,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                chunk_size=chunk_size,
                disable_recompute=disable_recompute,
                cp_context=cp_context,
            )
