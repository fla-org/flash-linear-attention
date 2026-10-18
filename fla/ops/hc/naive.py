# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Read out residual streams, accumulating in FP32 (FP64 for double inputs)."""
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return (x.to(dtype) * pre.to(dtype).unsqueeze(-1)).sum(-2).to(x.dtype)


def naive_hc_post(
    x: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write a branch into identity (iHC) or mixed (mHC) residual streams."""
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    residual = x.to(dtype)
    if mixing is not None:
        residual = (mixing.to(dtype).unsqueeze(-1) * residual.unsqueeze(-3)).sum(-2)
    return (residual + post.to(dtype).unsqueeze(-1) * branch.to(dtype).unsqueeze(-2)).to(x.dtype)


def naive_sinkhorn(logits: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Alternate column and row log normalizations and return the nonnegative mixing matrix."""
    value = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32)
    for _ in range(num_iters):
        value = value - value.logsumexp(dim=-2, keepdim=True)
        value = value - value.logsumexp(dim=-1, keepdim=True)
    return value.exp()
