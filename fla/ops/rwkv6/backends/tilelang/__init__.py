# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang backend for RWKV6 operations."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend
from fla.utils import find_spec_cached, has_usable_nvcc

_TILELANG_AVAILABLE = find_spec_cached("tilelang") is not None


class RWKV6TileLangBackend(BaseBackend):

    backend_type = "tilelang"
    package_name = "tilelang"
    env_var = "FLA_TILELANG"
    default_enable = False
    priority = 5

    @classmethod
    def is_available(cls) -> bool:
        return _TILELANG_AVAILABLE and has_usable_nvcc()

    def chunk_rwkv6_fwd_intra_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        gi: torch.Tensor,
        ge: torch.Tensor,
        u: torch.Tensor,
        scale: float,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[bool, str | None]:
        if cu_seqlens is not None or chunk_indices is not None:
            return False, "TileLang RWKV6 intra backend supports dense inputs only"
        if chunk_size != 64:
            return False, f"TileLang RWKV6 intra backend requires chunk_size=64, got {chunk_size}"
        if q.dtype != torch.bfloat16:
            return False, f"TileLang RWKV6 intra backend supports benchmark dtype bfloat16 only, got {q.dtype}"
        if k.dtype != q.dtype or u.dtype != q.dtype:
            return False, f"TileLang RWKV6 intra backend requires q/k/u dtype match, got {q.dtype}/{k.dtype}/{u.dtype}"
        if gi.dtype != torch.float32 or ge.dtype != torch.float32:
            return False, f"TileLang RWKV6 intra backend requires fp32 gi/ge, got {gi.dtype}/{ge.dtype}"
        if not q.is_cuda:
            return False, "TileLang RWKV6 intra backend requires CUDA tensors"
        if q.shape != k.shape or q.shape != gi.shape or q.shape != ge.shape:
            return False, "TileLang RWKV6 intra backend requires q, k, gi, and ge to have identical shapes"
        if q.ndim != 4:
            return False, f"TileLang RWKV6 intra backend expects q with rank 4, got {q.ndim}"
        B, T, H, K = q.shape
        if u.shape != (H, K):
            return False, f"TileLang RWKV6 intra backend requires u shape {(H, K)}, got {tuple(u.shape)}"
        if T % chunk_size != 0:
            return False, f"TileLang RWKV6 intra backend requires T divisible by {chunk_size}, got T={T}"
        if K != 64:
            return False, f"TileLang RWKV6 intra backend currently supports the D=64 benchmark bucket only, got K={K}"
        if B <= 0 or H <= 0:
            return False, f"TileLang RWKV6 intra backend requires positive B/H, got B={B}, H={H}"
        return True, None

    def chunk_rwkv6_fwd_intra(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        gi: torch.Tensor,
        ge: torch.Tensor,
        u: torch.Tensor,
        scale: float,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        from fla.ops.rwkv6.backends.tilelang.chunk_fwd_intra import (
            chunk_rwkv6_fwd_intra_tilelang,
        )
        return chunk_rwkv6_fwd_intra_tilelang(
            q=q,
            k=k,
            gi=gi,
            ge=ge,
            u=u,
            scale=scale,
            chunk_size=chunk_size,
        )
