# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch

from fla.backends import BaseBackend, register_backend
from fla.utils import IS_NVIDIA_HOPPER, TRITON_ABOVE_3_4_0, has_usable_nvcc


@register_backend('attn')
class AttnTileLangBackend(BaseBackend):
    backend_type = "tilelang"
    package_name = "tilelang"
    env_var = "FLA_TILELANG"
    # work around Hopper regressions with Triton 3.4+ (see #640).
    default_enable = IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0

    @classmethod
    def is_available(cls) -> bool:
        return super().is_available() and has_usable_nvcc()

    def parallel_attn_fwd_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_cumsum: torch.Tensor | None,
        sink_bias: torch.Tensor | None,
        scale: float,
        window_size: int | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[bool, str | None]:
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, f"TileLang backend does not support dtype {q.dtype}; fall back to Triton"
        return True, None

    def parallel_attn_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_cumsum: torch.Tensor | None,
        sink_bias: torch.Tensor | None,
        scale: float,
        window_size: int | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.ops.attn.backends.tilelang.parallel import parallel_attn_fwd_tilelang
        return parallel_attn_fwd_tilelang(
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            sink_bias=sink_bias,
            scale=scale,
            window_size=window_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    def parallel_attn_bwd_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        g_cumsum: torch.Tensor | None,
        lse: torch.Tensor,
        do: torch.Tensor,
        sink_bias: torch.Tensor | None = None,
        scale: float | None = None,
        window_size: int | None = None,
        chunk_size: int = 128,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[bool, str | None]:
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, f"TileLang backend does not support dtype {q.dtype}; fall back to Triton"
        return True, None

    def parallel_attn_bwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        g_cumsum: torch.Tensor | None,
        lse: torch.Tensor,
        do: torch.Tensor,
        sink_bias: torch.Tensor | None = None,
        scale: float | None = None,
        window_size: int | None = None,
        chunk_size: int = 128,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        from fla.ops.attn.backends.tilelang.parallel import parallel_attn_bwd_tilelang
        return parallel_attn_bwd_tilelang(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=g_cumsum,
            lse=lse,
            do=do,
            sink_bias=sink_bias,
            scale=scale,
            window_size=window_size,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )


__all__ = ['AttnTileLangBackend']
