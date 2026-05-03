# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.utils.cumsum import chunk_global_cumsum


class NormalizeWithZState(torch.autograd.Function):
    """Normalized linear-attention divide step with a hand-written backward.

    Forward computes the running key-cumsum z_t and divides ``o`` by ``<q_t, z_t>``.
    Backward mirrors the pattern used by simple_gla / gla / gated_delta_rule
    """

    @staticmethod
    def forward(ctx, o, q, k, scale, z_init, reverse, cu_seqlens):
        k_cum = chunk_global_cumsum(k, reverse=reverse, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
        if z_init is not None:
            if cu_seqlens is not None:
                seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
                z_init_b = z_init.squeeze(1).repeat_interleave(seq_lens, dim=0).unsqueeze(0)
            else:
                z_init_b = z_init
            k_cum = k_cum + z_init_b
        denom = (q * scale * k_cum).sum(-1, keepdim=True) + 1e-10
        o_out = o / denom
        if cu_seqlens is not None:
            idx = (cu_seqlens[:-1] if reverse else cu_seqlens[1:] - 1).to(torch.long)
            z_state_out = k_cum[0, idx].unsqueeze(1).contiguous()
        else:
            z_state_out = (k_cum[:, :1] if reverse else k_cum[:, -1:]).contiguous()

        ctx.save_for_backward(o, q, k_cum, denom)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        ctx.has_z_init = z_init is not None
        ctx.k_dtype = k.dtype
        return o_out, z_state_out

    @staticmethod
    def backward(ctx, do, dz_state):
        o, q, k_cum, denom = ctx.saved_tensors
        scale = ctx.scale
        reverse = ctx.reverse
        cu_seqlens = ctx.cu_seqlens

        d_o_in = do / denom
        ddenom = -(do * o).sum(-1, keepdim=True) / (denom * denom)

        dq = ddenom * scale * k_cum
        dk_cum = ddenom * scale * q

        if dz_state is not None:
            dk_cum = dk_cum.clone()
            if cu_seqlens is not None:
                idx = (cu_seqlens[:-1] if reverse else cu_seqlens[1:] - 1).to(torch.long)
                dk_cum[0].index_add_(0, idx, dz_state.squeeze(1).to(dk_cum.dtype))
            elif reverse:
                dk_cum[:, :1] = dk_cum[:, :1] + dz_state.to(dk_cum.dtype)
            else:
                dk_cum[:, -1:] = dk_cum[:, -1:] + dz_state.to(dk_cum.dtype)

        dz_init = None
        if ctx.has_z_init:
            if cu_seqlens is not None:
                seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
                seg_id = torch.repeat_interleave(
                    torch.arange(seq_lens.numel(), device=dk_cum.device), seq_lens,
                )
                flat = dk_cum[0]
                dz_init = torch.zeros(
                    seq_lens.numel(), flat.shape[1], flat.shape[2],
                    device=flat.device, dtype=flat.dtype,
                )
                dz_init.index_add_(0, seg_id, flat)
                dz_init = dz_init.unsqueeze(1)
            else:
                dz_init = dk_cum.sum(dim=1, keepdim=True)

        dk = chunk_global_cumsum(
            dk_cum, reverse=not reverse, cu_seqlens=cu_seqlens, output_dtype=ctx.k_dtype,
        )
        return d_o_in, dq, dk, None, dz_init, None, None


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
    return NormalizeWithZState.apply(o, q, k, scale, z_init, reverse, cu_seqlens)
