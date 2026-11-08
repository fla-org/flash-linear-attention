# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).


"""Shared packed-layout handling for SwiLA outputs and captured diagnostics."""
from __future__ import annotations

from collections.abc import Callable, Mapping

import torch


def validate_packed_boundaries(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> int:
    """Validate the recurrence's packed domain before frontend GPU kernels."""
    if cu_seqlens is None:
        return hidden_states.shape[0]
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be one-dimensional")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have dtype torch.int32 or torch.int64")
    if cu_seqlens.device != hidden_states.device:
        raise ValueError("cu_seqlens must be on the same device as hidden_states")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")
    if hidden_states.shape[0] != 1:
        raise ValueError("batch size must be 1 when using cu_seqlens; flatten variable-length inputs")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must describe at least one sequence")
    if cu_seqlens[0].item() != 0 or cu_seqlens[-1].item() != hidden_states.shape[1]:
        raise ValueError("cu_seqlens must start at 0 and end at the packed sequence length")
    if (cu_seqlens[1:] <= cu_seqlens[:-1]).any().item():
        raise ValueError("cu_seqlens must be strictly increasing; empty sequences are not supported")
    return cu_seqlens.numel() - 1


def validate_sequence_state_counts(
    states: Mapping[str, torch.Tensor | None],
    num_sequences: int,
) -> None:
    """Reject mismatched recurrent, prior and convolution caches before use."""
    for name, state in states.items():
        if state is None:
            continue
        if not isinstance(state, torch.Tensor) or state.ndim == 0:
            raise ValueError(f"{name} must be a tensor with a sequence dimension")
        if state.shape[0] != num_sequences:
            raise ValueError(
                f"{name} has {state.shape[0]} sequences but the input has {num_sequences}"
            )


def repad_packed_tensor(
    tensor: torch.Tensor | None,
    indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    *,
    pad_fn: Callable,
) -> torch.Tensor | None:
    """Restore ``[1, total, ...]`` with the caller's FLA padding implementation."""
    if tensor is None:
        return None
    return pad_fn(tensor.squeeze(0), indices, batch_size, seq_len)
