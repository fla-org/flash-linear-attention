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

import torch

from fla.ops.backends import BaseBackend


class TileLangBackend(BaseBackend):

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
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
        use_exp2: bool = False,
        transpose_state_layout: bool = False,
    ) -> tuple[bool, str | None]:
        if g is None:
            return False, "TileLang backend only supports gated case (g != None)"
        if g_gamma is not None:
            return False, "TileLang backend does not support g_gamma"
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
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
        use_exp2: bool = False,
        transpose_state_layout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        from fla.ops.common.backends.tilelang_kernels.chunk_bwd_dqkwg import (
            chunk_bwd_dqkwg_tilelang,
        )
        return chunk_bwd_dqkwg_tilelang(
            q=q, k=k, v=v, do=do, h=h, dh=dh,
            w=w, g=g, g_gamma=g_gamma, dv=dv,
            scale=scale, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size, chunk_indices=chunk_indices,
            use_exp2=use_exp2, transpose_state_layout=transpose_state_layout,
        )
