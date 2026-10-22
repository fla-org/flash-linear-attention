# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_sinkhorn(logits: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Apply column and row log normalizations in FP32."""
    x = logits.float()
    for _ in range(num_iters):
        x = x - x.logsumexp(dim=-2, keepdim=True)
        x = x - x.logsumexp(dim=-1, keepdim=True)
    return x.exp()
