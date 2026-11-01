# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_gate_output_correction(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Compute the RWKV7 output correction and gate with PyTorch."""
    correction = ((r * k * r_k.unsqueeze(0).unsqueeze(0)).sum(-1, keepdim=True) * v).view(o.shape)
    y = (o + correction) * g
    return y
