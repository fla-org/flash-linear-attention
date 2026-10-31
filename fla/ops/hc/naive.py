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
    """PyTorch reference for the HC/mHC/iHC readout in `fused_hc_pre`.

    Reduces `[..., N, D]` residual streams with `[..., N]` activated weights to a `[..., D]` branch input.
    Accumulates in FP32 and returns the stream dtype.
    """
    dtype = x.dtype
    x, pre = x.float(), pre.float()
    # [..., N, D] * [..., N, 1] -> [..., D]
    o = (x * pre.unsqueeze(-1)).sum(-2)
    return o.to(dtype)


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
    """
    dtype = x.dtype
    x, branch, post = x.float(), branch.float(), post.float()

    residual = x
    if mixing is not None:
        mixing = mixing.float()
        # [..., N, N, 1] * [..., 1, N, D] -> [..., N, D], reducing over input streams
        residual = (mixing.unsqueeze(-1) * x.unsqueeze(-3)).sum(-2)

    # [..., N, 1] * [..., 1, D] -> [..., N, D]
    update = post.unsqueeze(-1) * branch.unsqueeze(-2)
    return (residual + update).to(dtype)
