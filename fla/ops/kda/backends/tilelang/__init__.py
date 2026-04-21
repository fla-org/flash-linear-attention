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


class KDATileLangBackend(BaseBackend):

    backend_type = "tilelang"
    package_name = "tilelang"
    env_var = "FLA_TILELANG"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tilelang  # noqa: F401
            return True
        except ImportError:
            return False

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
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
        transpose_state_layout: bool = False,
    ) -> tuple[bool, str | None]:
        if v.shape[2] != k.shape[2]:
            return False, (
                "TileLang backend does not support GQA (v has more heads than k); "
                "use repeat_interleave on k/q to match v's head count, or fall back to Triton"
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
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
        transpose_state_layout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from fla.ops.kda.backends.tilelang.chunk_bwd_dqkg import (
            chunk_kda_bwd_wy_dqkg_fused_tilelang,
        )
        return chunk_kda_bwd_wy_dqkg_fused_tilelang(
            q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A,
            h=h, do=do, dh=dh, dv=dv,
            scale=scale, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size, chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state_layout,
        )
