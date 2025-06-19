# -*- coding: utf-8 -*-

import torch


class L2Wrap(torch.autograd.Function):
    r"""
    This class of penalty prevents the model from becoming overconfident,
    thereby mitigating precision loss in BF16.

    This version is memory-optimized by not storing the full logits tensor.
    """
    @staticmethod
    def forward(ctx, loss, logits, l2_penalty_factor=1e-4):
        """
        Forward pass for L2 penalty.
        Args:
            loss (torch.Tensor): The loss tensor.
            logits (torch.Tensor): Shape[B, T, V] The logits tensor.
            l2_penalty_factor (float): The factor for L2 penalty.
        """
        maxx, ids = torch.max(logits, dim=-1, keepdim=True)
        ctx.logits_shape = logits.shape
        factor = l2_penalty_factor / (logits.shape[0] * logits.shape[1])
        maxx = maxx * factor
        ctx.save_for_backward(maxx, ids)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        maxx, ids = ctx.saved_tensors
        glogits = torch.zeros(ctx.logits_shape, device=grad_output.device,
                              dtype=grad_output.dtype)
        glogits.scatter_(-1, ids, maxx)
        return grad_output, glogits, None


l2_warp = L2Wrap.apply
