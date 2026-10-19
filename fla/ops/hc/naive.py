# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

# pre/post precision follows the DeepSeek reference:
# https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6/tile_kernels/torch/mhc.py


def naive_hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Read out residual streams with FP32 accumulation."""
    dtype = x.dtype
    x, pre = x.float(), pre.float()
    o = (x * pre.unsqueeze(-1)).sum(-2)
    return o.to(dtype)


def naive_hc_post(
    x: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write a branch into identity (iHC) or mixed (mHC) residual streams."""
    dtype = x.dtype
    x, branch, post = x.float(), branch.float(), post.float()

    residual = x
    if mixing is not None:
        mixing = mixing.float()
        residual = (mixing.unsqueeze(-1) * x.unsqueeze(-3)).sum(-2)

    update = post.unsqueeze(-1) * branch.unsqueeze(-2)
    return (residual + update).to(dtype)
