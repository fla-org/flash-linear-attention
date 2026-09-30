# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F


def naive_stickbreaking_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    attend_current: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Reference stick-breaking attention, computed in fp32 log space.

    Query `i` gives each visible key `j` the weight `A_ij = sigmoid(z_ij) * prod_l (1 - sigmoid(z_il))`,
    where `z_ij = scale * q_i k_j` and `l` runs over the visible keys after `j`,
    that is `j < l < i`, or `j < l <= i` with `attend_current`.
    The nearest key breaks the stick first, and the weights of a query sum to at most 1.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`. GQA is applied if HQ is divisible by H.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        scale (float, Optional):
            Scale factor for the attention logits. Default: `1 / sqrt(K)`.
        attend_current (bool, Optional):
            Whether a query also attends to its own position. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
        rem (torch.Tensor):
            Stick left over by each query, `1 - sum_j A_ij`, of shape `[B, T, HQ]`.
    """
    B, T, HQ, K = q.shape
    H, V = k.shape[2], v.shape[-1]
    if H == 0 or HQ % H != 0:
        raise ValueError(f"The number of query heads ({HQ}) must be divisible by the number of key/value heads ({H}).")
    G = HQ // H
    if scale is None:
        scale = K ** -0.5

    dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    # [B, H, G, T, T]
    z = torch.einsum('bqhgd,bkhd->bhgqk', q.reshape(B, T, H, G, K), k) * scale

    i = torch.arange(T, device=q.device)
    mask = i[None, :] <= i[:, None] if attend_current else i[None, :] < i[:, None]
    # log(1 - beta) of every visible key, and 0 elsewhere so masked keys use none of the stick
    log_om_beta = F.logsigmoid(-z).masked_fill(~mask, 0.)
    # log(beta_ij) plus the stick used by the keys strictly between j and i
    log_att = F.logsigmoid(z) + log_om_beta.flip(-1).cumsum(-1).flip(-1) - log_om_beta
    att = log_att.masked_fill(~mask, float('-inf')).exp()

    o = torch.einsum('bhgqk,bkhd->bqhgd', att, v).reshape(B, T, HQ, V)
    rem = log_om_beta.sum(-1).exp().permute(0, 3, 1, 2).reshape(B, T, HQ)
    return o.to(dtype), rem.to(dtype)
