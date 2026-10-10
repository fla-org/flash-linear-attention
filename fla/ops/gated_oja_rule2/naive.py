# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_recurrent_gated_oja_rule2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Token-by-token reference of Oja2, written in plain PyTorch and differentiable.

    The recurrence acts on a slot memory `S` of shape `[K, M]`:

        S_t = S_{t-1} Diag(exp(gv_t)) (I - (b_t * v_t) v_t^T) + (c_t * k_t) v_t^T
        o_t = S_t^T q_t

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            slot code of shape `[B, T, H, M]`.
        gv (torch.Tensor):
            log-decay of shape `[B, T, H, M]`, in the natural-log base.
        b (torch.Tensor):
            per-slot erase gate of shape `[B, T, H, M]`.
        c (torch.Tensor):
            per-channel write gate of shape `[B, T, H, K]`.
        scale (float, Optional):
            Scale factor for the attention scores. If not provided, it defaults to `1 / sqrt(K)`.
        initial_state (torch.Tensor, Optional):
            Initial state of shape `[B, H, K, M]`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final state of shape `[B, H, K, M]`. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, M]`.
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, M]` if `output_final_state=True` else `None`.
    """
    dtype = v.dtype
    B, T, H, K, M = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, gv, b, c = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, gv, b, c))
    q = q * scale

    S = q.new_zeros(B, H, K, M)
    if initial_state is not None:
        S = S + initial_state.float()

    o = []
    for t in range(T):
        # [B, H, K, M]
        S = S * gv[:, :, t].exp().unsqueeze(-2)
        # read the state along the gated erase direction [B, H, K], then write (c * k) minus what was read
        erase = torch.einsum('bhkm,bhm->bhk', S, b[:, :, t] * v[:, :, t])
        write = c[:, :, t] * k[:, :, t] - erase
        S = S + write.unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        o.append(torch.einsum('bhkm,bhk->bhm', S, q[:, :, t]))

    o = torch.stack(o, dim=2).transpose(1, 2).contiguous().to(dtype)
    return o, (S if output_final_state else None)
