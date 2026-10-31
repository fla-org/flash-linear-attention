# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from einops import einsum

# pre/post precision follows the DeepSeek reference:
# https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6/tile_kernels/torch/mhc.py


def naive_hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for the HC/mHC/iHC readout in `fused_hc_pre`.

    Reduces `[..., N, D]` residual streams with `[..., N]` activated weights to a `[..., D]` branch input.
    Accumulates in FP32 and returns the stream dtype.

    Args:
        x (torch.Tensor):
            Residual streams of shape `[..., N, D]`, with N streams and D features per stream.
        pre (torch.Tensor):
            Activated readout weights of shape `[..., N]`, with the same leading dimensions as `x`.

    Returns:
        torch.Tensor:
            Weighted readout of shape `[..., D]`, accumulated in FP32 and returned in `x.dtype`.
    """
    return einsum(x.float(), pre.float(), '... n d, ... n -> ... d').to(x.dtype)


def naive_hc_post(
    x: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch reference for the residual update in `fused_hc_post`.

    HC mixes N residual streams before adding the branch output weighted by `post`.
    mHC uses a caller-supplied doubly stochastic mixing matrix; iHC uses `mixing=None` for the identity transition.
    Accumulates in FP32 and returns `[..., N, D]` in the stream dtype.

    Args:
        x (torch.Tensor):
            Residual streams of shape `[..., N, D]`, with N streams and D features per stream.
        branch (torch.Tensor):
            Attention or MLP branch output of shape `[..., D]`, with the same leading dimensions as `x`.
        post (torch.Tensor):
            Activated write weights of shape `[..., N]`, with the same leading dimensions as `x`.
        mixing (torch.Tensor, Optional):
            Residual mixing matrices of shape `[..., N, N]`, indexed by output stream then input stream.
            Use a doubly stochastic matrix for mHC or `None` for identity mixing in iHC. Default: `None`.

    Returns:
        torch.Tensor:
            Updated residual streams of shape `[..., N, D]`, accumulated in FP32 and returned in `x.dtype`.
    """
    residual = x.float()
    if mixing is not None:
        residual = einsum(mixing.float(), residual, '... n m, ... m d -> ... n d')
    update = einsum(post.float(), branch.float(), '... n, ... d -> ... n d')
    return (residual + update).to(x.dtype)
