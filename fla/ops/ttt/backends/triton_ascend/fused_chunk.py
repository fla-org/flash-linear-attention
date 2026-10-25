# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend delegation for the fused-chunk TTT linear op."""

from __future__ import annotations

import torch

from fla.ops.ttt.chunk import ChunkTTTLinearFunction, norm_residual


def fused_chunk_ttt_linear_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float | None = None,
    eps: float = 1e-6,
    chunk_size: int = 16,
    initial_state: torch.Tensor | None = None,
    initial_state_bias: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    # Mirror of the mainline validation and defaults (fla/ops/ttt/fused_chunk.py).
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[-1] == v.shape[-1], "DK must equal to DV."
    if isinstance(eta, float):
        eta = torch.full_like(q[:, :, :, :1], eta)
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "Scale must be positive."
    # The fused backward dh kernel exceeds the NPU Unified Buffer budget and
    # its tiles cannot be shrunk, so delegate to the chunked implementation.
    o, final_state, final_state_bias = ChunkTTTLinearFunction.apply(
        q,
        k,
        v,
        w,
        b,
        chunk_size,
        eta,
        scale,
        eps,
        initial_state,
        initial_state_bias,
        output_final_state,
        cu_seqlens,
        None,  # cu_seqlens_cpu
    )
    o = norm_residual(o, w, b, eps)
    return o, final_state, final_state_bias


__all__ = ['fused_chunk_ttt_linear_npu']
