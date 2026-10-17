# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2


@triton.jit
def gdn2_local_log_cumsum(g, start, end, o_k, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, REVERSE: tl.constexpr = False):
    """Return base-2 prefixes or exclusive suffixes from natural-log gates."""
    # the reverse scan excludes the source token's gate without subtracting a rounded prefix
    o_t = start + tl.arange(0, BT) + (1 if REVERSE else 0)
    b_g = tl.load(g + o_t[:, None] * H * K + o_k[None, :], (o_t[:, None] < end) & (o_k[None, :] < K), other=0).to(tl.float32)
    return tl.cumsum(b_g, axis=0, reverse=REVERSE) * 1.4426950408889634


@triton.jit(do_not_specialize=['T'])
def gdn2_chunk_gate_kernel(
    g,
    out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    dg=None,
):
    i_tk = tl.program_id(0).to(tl.int64)
    i_t, i_k = i_tk // tl.cdiv(K, BK), i_tk % tl.cdiv(K, BK)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        T = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    else:
        bos = i_b * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_t = i_t * BT + tl.arange(0, BT)
    offsets = ((bos + o_t[:, None]) * H + i_h) * K + o_k[None, :]
    mask = (o_t[:, None] < T) & (o_k[None, :] < K)
    if dg is None:
        value = gdn2_local_log_cumsum(g + (bos * H + i_h) * K, i_t * BT, min(i_t * BT + BT, T), o_k, H, K, BT, True)
    else:
        b_dg = tl.load(dg + offsets, mask, other=0).to(tl.float32)
        b_g = tl.load(g + offsets, mask, other=0).to(tl.float32)
        value = tl.cumsum(b_dg, axis=0, reverse=True)
        # an underflowed decay has zero derivative, even when prefix-gradient cancellation leaves roundoff
        value = tl.where(exp2(b_g * 1.4426950408889634) == 0, 0., value)
    tl.store(out + offsets, value, mask)


def gdn2_chunk_gate(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    dg: torch.Tensor | None = None,
):
    """Compute log suffixes or reduce prefix gradients to token-gate gradients."""
    B, T, H, K = g.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, chunk_size) if chunk_indices is None else len(chunk_indices)
    BK = min(32, triton.next_power_of_2(K))
    out = torch.empty_like(g, dtype=torch.float32)
    gdn2_chunk_gate_kernel[(NT * triton.cdiv(K, BK), B * H)](
        g=g,
        out=out,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=chunk_size,
        BK=BK,
        IS_VARLEN=cu_seqlens is not None,
        dg=dg,
    )
    return out
