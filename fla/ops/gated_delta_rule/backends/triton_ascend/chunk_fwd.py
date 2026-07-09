# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_gated_delta_rule_fwd_intra adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.utils import prepare_chunk_indices, solve_tril
from fla.utils import input_guard


@input_guard
def chunk_gated_delta_rule_fwd_intra_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if chunk_size not in (16, 32, 64):
        raise ValueError(f'`chunk_size` must be 16, 32, or 64, got {chunk_size}.')

    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    # Unfused kkt + solve_tril path to stay within UB budget.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, A
