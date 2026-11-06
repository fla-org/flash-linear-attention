# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


class L2Wrap(torch.autograd.Function):
    r"""
    This class of penalty prevents the model from becoming overconfident,
    thereby mitigating precision loss in BF16.

    This version is memory-optimized by not storing the full logits tensor.
    """
    @staticmethod
    def forward(
        ctx,
        loss: torch.Tensor,
        logits: torch.Tensor,
        l2_penalty_factor: float = 1e-4,
    ) -> torch.Tensor:
        """
        Args:
            loss (torch.Tensor):
                The already-reduced (scalar) loss to wrap.
            logits (torch.Tensor):
                The logits of shape `[B, T, V]`.
            l2_penalty_factor (float, Optional):
                The strength of the L2 penalty on the max logit. Default: 1e-4.
        """
        maxx, ids = torch.max(logits, dim=-1, keepdim=True)
        ctx.logits_shape = logits.shape
        factor = l2_penalty_factor / (logits.shape[0] * logits.shape[1])
        maxx = maxx * factor
        ctx.save_for_backward(maxx, ids)
        return loss

    @staticmethod
    def backward(ctx, dloss: torch.Tensor):
        maxx, ids = ctx.saved_tensors
        dlogits = torch.zeros(ctx.logits_shape, device=dloss.device, dtype=maxx.dtype)
        # scale before scattering to avoid another full-size logits allocation
        dlogits.scatter_(-1, ids, (maxx * dloss).to(maxx.dtype))
        return dloss, dlogits, None


l2_warp = L2Wrap.apply
