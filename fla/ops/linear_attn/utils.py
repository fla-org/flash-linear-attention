# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.utils.cumsum import chunk_global_cumsum


def normalize_with_z_state(
    o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    z_init: torch.Tensor | None,
    reverse: bool,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply running normalization Z_t = sum_{s<=t} k_s (or sum_{s>=t} for reverse).

    z_init carries the cumulative key from prior chunks; the returned z_state is the
    boundary needed to chain into the next chunk. Varlen path broadcasts z_init and
    extracts the boundary per-sequence.
    """
    k_cum = chunk_global_cumsum(k, reverse=reverse, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    if z_init is not None:
        if cu_seqlens is not None:
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
            z_init = z_init.squeeze(1).repeat_interleave(seq_lens, dim=0).unsqueeze(0)
        k_cum = k_cum + z_init
    o = o / ((q * scale * k_cum).sum(-1, keepdim=True) + 1e-10)
    if cu_seqlens is not None:
        idx = (cu_seqlens[:-1] if reverse else cu_seqlens[1:] - 1).to(torch.long)
        z_state_out = k_cum[0, idx].unsqueeze(1).contiguous()
    else:
        z_state_out = k_cum[:, :1] if reverse else k_cum[:, -1:]
    return o, z_state_out
