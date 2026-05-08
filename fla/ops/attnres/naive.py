# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import einsum


def naive_attnres(
    query: torch.Tensor,
    residuals: torch.Tensor | Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
    return_residuals: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""
    Apply AttnRes residual aggregation.

    AttnRes normalizes each residual source with RMSNorm, scores it against
    `query`, applies softmax over the residual-source dimension, and returns
    the weighted sum of residual sources.
    See `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.

    Args:
        query (torch.Tensor):
            Per-layer pseudo-query of shape `[D]` or `[D, 1]`, where `D` is
            the hidden size.
        residuals (torch.Tensor or Sequence[torch.Tensor]):
            Residual sources of shape `[L, ..., D]`, or a sequence of tensors
            each with shape `[..., D]`, where `L` is the number of residual
            sources.
        rms_weight (torch.Tensor):
            RMSNorm scale for key normalization of shape `[D]`.
        rms_eps (float):
            RMSNorm epsilon. Default: `1e-6`.
        scale (float):
            Scale factor applied to AttnRes logits before softmax. Default: `1.0`.
        return_weights (bool):
            Whether to return depth softmax probabilities. Default: `False`.
        return_residuals (bool):
            Whether to return the stacked residual tensor. Useful when the
            input was a list and the caller wants the materialized
            `[L, ..., D]` tensor without re-stacking. Default: `False`.

    Returns:
        o (torch.Tensor):
            Mixed residual of shape `[..., D]`.
        p (torch.Tensor):
            Depth softmax probabilities of shape `[L, ...]` if
            `return_weights=True`, otherwise not returned.
        residuals (torch.Tensor):
            Stacked residuals of shape `[L, ..., D]` if `return_residuals=True`,
            otherwise not returned.
    """
    output_shape = None
    if isinstance(residuals, Sequence) and not isinstance(residuals, torch.Tensor):
        if len(residuals) == 0:
            raise ValueError("residuals must contain at least one source")
        output_shape = residuals[0].shape
        D = output_shape[-1]
        residuals = torch.stack(tuple(residual.view(-1, D) for residual in residuals), dim=0)

    v = residuals.float()
    k = F.rms_norm(v, (residuals.shape[-1],), rms_weight.flatten().float(), rms_eps)
    p = (einsum(k, query.flatten().float() * scale, "l ... d, d -> l ...")).softmax(dim=0)
    o = einsum(p, v, "l ..., l ... d -> ... d").to(residuals.dtype)
    if output_shape is not None:
        o = o.view(output_shape)

    outputs = [o]
    if return_weights:
        if output_shape is not None:
            p = p.view(residuals.shape[0], *output_shape[:-1])
        outputs.append(p)
    if return_residuals:
        if output_shape is not None:
            residuals = residuals.view(residuals.shape[0], *output_shape)
        outputs.append(residuals)
    return tuple(outputs) if len(outputs) > 1 else o


__all__ = ["naive_attnres"]
