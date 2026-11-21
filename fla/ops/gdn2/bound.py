# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def bound_symmetric_key(u: torch.Tensor) -> torch.Tensor:
    """Cap a symmetric erase key with headroom for its final dtype cast.

    Use the returned key with an all-ones erase gate and kernel normalization disabled.
    The target squared norm is 1.99. Near the boundary, this changes erase and write strength.
    It does not bound the generic channel-wise erase operator or guarantee kernel accuracy.

    Args:
        u (torch.Tensor):
            Already-transformed keys with key channels on the last axis.
            Inputs must be finite and have a finite squared norm in the accumulation dtype.
            Supported dtypes are float16, bfloat16, float32, and float64.

    Returns:
        torch.Tensor:
            Keys with the input shape and dtype. Autograd differentiates through the radial cap.
    """
    if u.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("Expected float16, bfloat16, float32, or float64 keys.")
    acc = u.to(torch.promote_types(u.dtype, torch.float32))
    # reserve final-cast rounding headroom and reduction slack
    ceiling = 1.99 / (1 + torch.finfo(u.dtype).eps / 2) ** 2 * (1 - 1e-6)
    norm2 = acc.square().sum(dim=-1, keepdim=True)
    scale = (norm2 / ceiling).clamp_min(1).rsqrt()
    return (acc * scale).to(u.dtype)
