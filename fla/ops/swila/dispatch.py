# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Recurrent route eligibility: metadata only, never kernel launches.

General versus split is a decomposition choice; fixed launch schedules are a
separate choice. These predicates preserve the pre-reorganization restrictions.
"""


from __future__ import annotations

import torch


def _use_dense_d64e4_key_deferred_readout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    log_decay: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
) -> bool:
    """Select the SM90-only deferred-readout scheduling specialization."""
    return (
        log_decay is not None
        and q.ndim == 4
        and q.shape[1] == 4096
        and _is_nvidia_sm90_tensor(q)
        and _use_dense_d64e4_key_bwd_fastpath(
            q,
            k,
            v,
            beta,
            pi_q,
            pi_k,
            log_decay,
            cu_seqlens,
        )
    )


def _is_nvidia_sm90_tensor(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` resides on an exact NVIDIA SM90 device."""
    return bool(
        tensor.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(tensor.device) == (9, 0)
    )


def _use_dense_d64e4_key_bwd_fastpath(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    log_decay: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
) -> bool:
    """Select the isolated production backward for its exact BF16 shape."""
    activations = (q, k, v, beta, pi_q, pi_k)
    if (
        cu_seqlens is not None
        or any(
            tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            for tensor in activations
        )
        or log_decay.dtype != torch.float32
        or not log_decay.is_cuda
        or not log_decay.is_contiguous()
        or q.ndim != 4
    ):
        return False

    prefix = tuple(q.shape[:-1])
    return (
        tuple(q.shape) == (*prefix, 64)
        and tuple(k.shape) == (*prefix, 64)
        and tuple(v.shape) == (*prefix, 64)
        and tuple(beta.shape) == (*prefix, 64, 4)
        and tuple(pi_q.shape) == tuple(beta.shape)
        and tuple(pi_k.shape) == tuple(beta.shape)
        and tuple(log_decay.shape) == tuple(q.shape)
    )


def _use_temporal_dense_d64e4_key_bwd_fastpath(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    log_decay: torch.Tensor | None,
    gate_kv: torch.Tensor | None,
    gate_q: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    """Select the exact dense FineWeb temporal backward specialization."""
    activations = (q, k, v, beta, pi_q, pi_k, gate_kv, gate_q)
    if (
        cu_seqlens is not None
        or log_decay is None
        or log_decay.dtype != torch.float32
        or not log_decay.is_cuda
        or not log_decay.is_contiguous()
        or any(
            tensor is None
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            for tensor in activations
        )
        or q.ndim != 4
    ):
        return False

    prefix = tuple(q.shape[:-1])
    gate_shape = (*prefix, 64)
    return (
        tuple(q.shape) == (*prefix, 64)
        and tuple(k.shape) == (*prefix, 64)
        and tuple(v.shape) == (*prefix, 64)
        and tuple(beta.shape) == (*prefix, 64, 4)
        and tuple(pi_q.shape) == tuple(beta.shape)
        and tuple(pi_k.shape) == tuple(beta.shape)
        and tuple(log_decay.shape) == tuple(q.shape)
        and tuple(gate_kv.shape) == gate_shape
        and tuple(gate_q.shape) == gate_shape
    )
