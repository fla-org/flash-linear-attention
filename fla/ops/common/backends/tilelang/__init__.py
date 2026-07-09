# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang backend for common chunk operations.

Enabled by default on Hopper (sm90+) with Triton >= 3.4.0 to work around
hardware-specific regressions (see #640). Can also be forced via FLA_TILELANG=1.
"""

from __future__ import annotations

import os

import torch

from fla.ops.backends import BaseBackend
from fla.utils import IS_NVIDIA_HOPPER, TRITON_ABOVE_3_4_0, find_spec_cached

_TILELANG_AVAILABLE = find_spec_cached("tilelang") is not None


class TileLangBackend(BaseBackend):

    backend_type = "tilelang"
    package_name = "tilelang"
    env_var = "FLA_TILELANG"

    @classmethod
    def is_available(cls) -> bool:
        return _TILELANG_AVAILABLE

    @classmethod
    def is_enabled(cls) -> bool:
        # Explicit opt-in / opt-out always wins.
        val = os.environ.get(cls.env_var)
        if val is not None:
            return val != "0"
        # Default on only where the Triton path is known to be broken:
        # Hopper (sm90) with Triton >= 3.4.0 (see #640). Everywhere else the
        # Triton backend stays the default unless the user forces FLA_TILELANG=1.
        return IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0

    def chunk_bwd_dqkwg_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        h: torch.Tensor,
        dh: torch.Tensor,
        w: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        g_gamma: torch.Tensor | None = None,
        dv: torch.Tensor | None = None,
        scale: float | None = None,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[bool, str | None]:
        if g is None:
            return False, "TileLang backend only supports gated case (g != None)"
        if g_gamma is not None:
            return False, "TileLang backend does not support g_gamma"
        if v.shape[2] % k.shape[2] != 0:
            return False, (
                f"TileLang backend requires num_v_heads (HV={v.shape[2]}) to be divisible by "
                f"num_qk_heads (H={k.shape[2]}); HV % H must be 0 for GVA"
            )
        if h.dtype != q.dtype:
            return False, (
                f"TileLang backend requires h.dtype == q.dtype (got h={h.dtype}, q={q.dtype}); "
                "e.g. simple_gla's bwd keeps h/dh in fp32 for h·dh reduction precision"
            )
        return True, None

    def chunk_bwd_dqkwg(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        h: torch.Tensor,
        dh: torch.Tensor,
        w: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        g_gamma: torch.Tensor | None = None,
        dv: torch.Tensor | None = None,
        scale: float | None = None,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        from fla.ops.common.backends.tilelang.chunk_bwd import (
            chunk_bwd_dqkwg_tilelang,
        )
        return chunk_bwd_dqkwg_tilelang(
            q=q,
            k=k,
            v=v,
            do=do,
            h=h,
            dh=dh,
            w=w,
            g=g,
            g_gamma=g_gamma,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )

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
        from fla.ops.common.backends.tilelang.parallel_attn_fwd import (
            parallel_attn_fwd_tilelang,
        )
        return parallel_attn_fwd_tilelang(
            q=q, k=k, v=v, g_cumsum=g_cumsum, sink_bias=sink_bias,
            scale=scale, window_size=window_size, cu_seqlens=cu_seqlens,
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
        from fla.ops.common.backends.tilelang.parallel_attn_bwd import (
            parallel_attn_bwd_tilelang,
        )
        return parallel_attn_bwd_tilelang(
            q=q, k=k, v=v, o=o, g_cumsum=g_cumsum, lse=lse, do=do,
            sink_bias=sink_bias, scale=scale, window_size=window_size,
            chunk_size=chunk_size, cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
