# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch

from fla.backends import TileLangBackend, register
from fla.utils import IS_NVIDIA_HOPPER, TRITON_ABOVE_3_4_0


@register('common')
class CommonTileLangBackend(TileLangBackend):
    # work around Hopper regressions with Triton 3.4+ (see #640).
    default_enable = IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0

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
        from fla.ops.common.backends.tilelang.chunk_bwd import chunk_bwd_dqkwg_tilelang
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
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )


__all__ = ['CommonTileLangBackend']
