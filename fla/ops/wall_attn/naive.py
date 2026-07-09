# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Wall attention, contributed by Tilde Research (Timor Averbuch, Dhruv Pai).
"""Eager PyTorch reference for Wall attention (correctness oracle)."""

import torch

from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_global_cumsum


def naive_wall_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    sink_bias: torch.Tensor | None = None,
    g_scalar: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Eager Wall forward (exact logits): per pair :math:`(i,j)`,
    :math:`s_{ij} = \sum_n q_{in} k_{jn} \exp_2(P_{in}-P_{jn}) \cdot \mathrm{scale}\cdot\mathrm{RCP\_LN2}`
    with :math:`P` the prefix cumsum of ``g`` in log_2 (same as the Triton path)."""
    if g.dim() != 4 or g.shape != (*q.shape[:-1], q.shape[-1]):
        raise ValueError("`g` must be [B, T, HQ, K] matching `q`")
    G = q.shape[2] // k.shape[2]
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("HQ must be divisible by H (GQA)")

    P = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, scale=RCP_LN2)
    k_exp = k.repeat_interleave(G, dim=2)
    B, T, HQ, _K = q.shape
    diff = P.unsqueeze(2).float() - P.unsqueeze(1).float()
    scores = (
        q.unsqueeze(2).float() * k_exp.unsqueeze(1).float() * torch.exp2(diff)
    ).sum(-1).permute(0, 3, 1, 2).contiguous()

    i_idx = torch.arange(T, device=q.device, dtype=torch.long).view(1, T, 1)
    j_idx = torch.arange(T, device=q.device, dtype=torch.long).view(1, 1, T)
    valid = j_idx <= i_idx
    if window_size is not None:
        valid = valid & (i_idx - j_idx < window_size)
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("reference varlen expects batch size 1")
        seg = torch.zeros(T, dtype=torch.long, device=q.device)
        for s in range(int(cu_seqlens.numel()) - 1):
            a = int(cu_seqlens[s].item())
            b = int(cu_seqlens[s + 1].item())
            seg[a:b] = s
        seg_i = seg.view(1, T, 1)
        seg_j = seg.view(1, 1, T)
        valid = valid & (seg_i == seg_j)

    valid_hq = valid.unsqueeze(1).expand(B, HQ, T, T)
    scores = scores.masked_fill(~valid_hq, float("-inf")) * (scale * RCP_LN2)

    if g_scalar is not None:
        c = chunk_global_cumsum(g_scalar, cu_seqlens=cu_seqlens, scale=RCP_LN2)
        c_hq = c.permute(0, 2, 1).float()
        scores = scores + c_hq.unsqueeze(-1) - c_hq.unsqueeze(-2)

    m = scores.amax(dim=-1, keepdim=True)
    m_stable = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    p = torch.exp2(scores - m_stable)
    den = p.sum(dim=-1)
    if sink_bias is not None:
        sink_l2 = (sink_bias * RCP_LN2).view(1, HQ, 1)
        den = den + torch.exp2(sink_l2 - m_stable.squeeze(-1))
    w = p / den.unsqueeze(-1)
    v_h = v.repeat_interleave(G, dim=2).permute(0, 2, 1, 3).float().contiguous()
    o = torch.matmul(w, v_h).transpose(1, 2).contiguous()
    return o
