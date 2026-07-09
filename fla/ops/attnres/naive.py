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
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""
    Apply AttnRes residual aggregation.

    AttnRes normalizes each residual source with RMSNorm, scores it against `query`, applies softmax over the
    residual-source dimension, and returns the weighted sum of residual sources.
    See `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.

    Args:
        query (torch.Tensor):
            Per-layer pseudo-query of shape `[D]` or `[D, 1]`, where `D` is the hidden size.
        residuals (Sequence[torch.Tensor]):
            Non-empty sequence of same-dtype, same-`D` residual sources, each of shape `[..., D]`.
        rms_weight (torch.Tensor):
            RMSNorm scale for key normalization of shape `[D]`.
        output_rms_weight (torch.Tensor, optional):
            If set, an extra RMSNorm with this weight is applied to the mixed residual before returning, fusing the
            prenorm that would otherwise follow the AttnRes call (e.g. `attn_norm` / `mlp_norm`). Default: `None`.
        rms_eps (float):
            RMSNorm epsilon (also used for `output_rms_weight` when set). Default: `1e-6`.
        scale (float):
            Scale factor applied to AttnRes logits before softmax. Default: `1.0`.
        return_weights (bool):
            Whether to return depth softmax probabilities. Default: `False`.

    Returns:
        o (torch.Tensor):
            Mixed residual of shape `[..., D]`.
        p (torch.Tensor):
            Depth softmax probabilities of shape `[L, ...]` if `return_weights=True`, otherwise not returned.
    """
    if len(residuals) == 0:
        raise ValueError("residuals must contain at least one source")

    output_shape = residuals[0].shape
    D = output_shape[-1]
    # the stack here is part of the *reference* impl, not the API; it lets einsum-style math run on one tensor while autograd
    # still routes per-source gradients back to each leaf through the inner views.
    stacked = torch.stack(tuple(residual.view(-1, D) for residual in residuals), dim=0)

    # all math runs in fp32 end-to-end; the final downcast back to the residual dtype happens once, just before returning.
    v = stacked.float()
    k = F.rms_norm(v, (D,), rms_weight.flatten().float(), rms_eps)
    p = (einsum(k, query.flatten().float() * scale, "l ... d, d -> l ...")).softmax(dim=0)
    o = einsum(p, v, "l ..., l ... d -> ... d")
    o = o.view(output_shape)

    if output_rms_weight is not None:
        o = F.rms_norm(o, (D,), output_rms_weight.float(), rms_eps)

    o = o.to(stacked.dtype)

    if return_weights:
        p = p.view(len(residuals), *output_shape[:-1])
        return o, p
    return o


__all__ = ["naive_attnres"]
