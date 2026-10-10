# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend NPU realization of `fused_chunk_simple_gla`.

`fused_chunk_bwd_kernel` cannot be lowered correctly by Triton-Ascend: `BT=64`
faults with an unaligned vector UB access (AICore 507015) and `BT=32`
miscompiles `dk`. The NPU path realizes the same semantics with the chunk
decomposition.
"""

from __future__ import annotations

import torch


def fused_chunk_simple_gla_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor = None,
    g_gamma: torch.Tensor = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.simple_gla.chunk import chunk_simple_gla

    if scale is None:
        scale = k.shape[-1] ** -0.5
    return chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
