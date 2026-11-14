# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_block_causal_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    block_size: int = 64,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quadratic reference with an explicit block-causal mask, `o_t = scale * q_t^T S_{e(t)}`."""
    dtype = q.dtype
    B, T, H, K = q.shape
    if scale is None:
        scale = K ** -0.5
    q, k, v = q.float(), k.float(), v.float()
    # G_t: cumulative log decay; e(t): last position of the block of t
    G = g.float().cumsum(1) if g is not None else q.new_zeros(B, T, H)
    pos = torch.arange(T, device=q.device)
    end = torch.clamp((pos // block_size + 1) * block_size, max=T) - 1
    mask = pos[None, :] <= end[:, None]
    # [B, H, T, S]: exp(G_{e(t)} - G_s) for s <= e(t)
    G_end = G[:, end].transpose(1, 2)
    decay = (G_end[..., :, None] - G.transpose(1, 2)[..., None, :]).masked_fill(~mask, float('-inf')).exp()
    attn = torch.einsum('bthk,bshk->bhts', q, k) * decay
    o = torch.einsum('bhts,bshv->bthv', attn, v)
    if initial_state is not None:
        o = o + torch.einsum('bthk,bhkv->bthv', q * G_end.transpose(1, 2)[..., None].exp(), initial_state.float())
    o = o * scale
    ht = None
    if output_final_state:
        w = (G[:, -1:] - G).exp()
        ht = torch.einsum('bshk,bshv->bhkv', k * w[..., None], v)
        if initial_state is not None:
            ht = ht + G[:, -1].exp()[..., None, None] * initial_state.float()
    return o.to(dtype), ht
