# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang backend for DPLR operations."""

from __future__ import annotations

import functools

import torch

from fla.ops.backends import BaseBackend
from fla.utils import find_spec_cached, has_usable_nvcc

_TILELANG_AVAILABLE = find_spec_cached("tilelang") is not None


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _smem_optin_bytes(device_index: int) -> int:
    props = torch.cuda.get_device_properties(device_index)
    return int(getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block))


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
        **kwargs,
    ) -> tuple[bool, str | None]:
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
        chunk_size = 16 if chunk_size is None else chunk_size
        if chunk_size == 16 and k.shape[-1] == 128:
            # measured ~0.5x vs Triton (the non-vectorized A-stage and the
            # 2-warp h+o path at BT=16 do not pay off at K=128)
            return False, "TileLang backend is slower than Triton at chunk_size 16 with head dim 128; fall back to Triton"
        if chunk_size == 64:
            # the intra backward at BT=64 needs 131200B (K=64) or 148480B
            # (K=128) of shared memory per block; smaller caps cannot run it
            smem_need = 131200 if k.shape[-1] <= 64 else 148480
            if _smem_optin_bytes(q.device.index or 0) < smem_need:
                return False, (
                    f"TileLang backend supports chunk_size 64 only on devices with "
                    f">= {smem_need}B shared memory per block "
                    f"(got {_smem_optin_bytes(q.device.index or 0)}B); fall back to Triton"
                )
        if chunk_size not in (16, 32, 64):
            return False, f"TileLang backend supports chunk_size 16/32/64, got {chunk_size}; fall back to Triton"
        if not q.is_cuda:
            return False, "TileLang backend is CUDA-only; fall back to Triton"
        # The fused h+o state pass parallelizes only over (V/BV, N, H) blocks
        # and walks chunks serially, so it loses to the split Triton kernels
        # when the grid underfills the GPU. Measured crossover on PRO 6000 /
        # H100 class parts is around half the SM count.
        bv = 64 if v.shape[-1] <= 64 else 32
        n_seqs = len(cu_seqlens) - 1 if cu_seqlens is not None else q.shape[0]
        grid = n_seqs * q.shape[2] * ((v.shape[-1] + bv - 1) // bv)
        sm = _sm_count(q.device.index or 0)
        if grid < sm // 2:
            return False, (
                f"TileLang backend is slower than Triton on small grids (N*H*(V/BV)={grid} "
                f"< {sm // 2} SMs/2); fall back to Triton"
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
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.ops.generalized_delta_rule.dplr.backends.tilelang.chunk import (
            chunk_dplr_delta_rule_tilelang,
        )
        return chunk_dplr_delta_rule_tilelang(
            q=q, k=k, v=v, a=a, b=b, gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            safe_gate=safe_gate,
            chunk_size=chunk_size,
            disable_recompute=disable_recompute,
            cp_context=cp_context,
        )
