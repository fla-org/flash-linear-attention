# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2
from fla.utils import get_multiprocessor_count


@triton.jit
def gdn2_bwd_diagonal(q, k, b, g_raw, dAqk, dAkk, start, T, o_k,
                      H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    o_i = tl.arange(0, BC)
    o_t = start + o_i
    mask = (o_t[:, None] < T) & (o_k[None, :] < K)
    offsets = o_t[:, None] * H * K + o_k[None, :]
    b_q = tl.load(q + offsets, mask=mask, other=0).to(tl.float32)
    b_k = tl.load(k + offsets, mask=mask, other=0).to(tl.float32)
    b_b = tl.load(b + offsets, mask=mask, other=0).to(tl.float32)
    b_g = tl.load(g_raw + offsets + H * K,
                  mask=(o_t[:, None] + 1 < T) & (o_k[None, :] < K), other=0).to(tl.float32)
    BK: tl.constexpr = o_k.shape[0]
    BR: tl.constexpr = 2
    dq = tl.zeros([BC, BK], tl.float32)
    dk = tl.zeros([BC, BK], tl.float32)
    dkt = tl.zeros([BC, BK], tl.float32)
    output_indices = tl.broadcast_to((o_i % BR)[:, None], (BC, BK))
    # tile destination rows so scan and reduction temporaries fit Ascend UB
    for block in range(tl.cdiv(BC, BR)):
        o_r = block * BR + tl.arange(0, BR)
        row_indices = tl.broadcast_to(o_r[:, None], (BR, BK))
        b_qr = tl.gather(b_q, row_indices, axis=0)
        b_kr = tl.gather(b_k, row_indices, axis=0)
        b_br = tl.gather(b_b, row_indices, axis=0)
        # the reverse scan includes only gates j+1 through destination r
        interval = tl.where(o_r[:, None, None] > o_i[None, :, None], b_g[None, :, :], 0.)
        decay = exp2(tl.cumsum(interval, axis=1, reverse=True) * 1.4426950408889634)
        valid = (o_r[:, None] >= o_i[None, :]) & (start + o_r[:, None] < T) & (o_t[None, :] < T)
        offsets_a = (start + o_r[:, None]) * H * BT + start % BT + o_i[None, :]
        b_daq = tl.load(dAqk + offsets_a, mask=valid, other=0).to(tl.float32)
        b_dak = tl.load(dAkk + offsets_a, mask=valid, other=0).to(tl.float32)
        dq_rows = tl.sum(b_daq[:, :, None] * b_k[None, :, :] * decay, axis=1)
        dk_rows = tl.sum(b_dak[:, :, None] * b_k[None, :, :] * decay, axis=1)
        dq = tl.where((o_i // BR == block)[:, None], tl.gather(dq_rows, output_indices, axis=0), dq)
        dk = tl.where((o_i // BR == block)[:, None], tl.gather(dk_rows, output_indices, axis=0), dk)
        dkt += tl.sum(b_daq[:, :, None] * b_qr[:, None, :] * decay, axis=0)
        dkt += tl.sum(b_dak[:, :, None] * (b_kr * b_br)[:, None, :] * decay, axis=0)
    return dq, dk, dkt


@triton.jit(do_not_specialize=['B', 'T', 'NT'])
def gdn2_bwd_kernel_diagonal(
    q, k, b, g_raw, dAqk, dAkk, dq, dk, dkt, cu_seqlens, chunk_indices, B, T, NT,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    NK: tl.constexpr = tl.cdiv(K, BK)
    NC: tl.constexpr = tl.cdiv(BT, BC)
    count = tl.cast(B, tl.int64) * H * NT * NC * NK
    for task in range(tl.cast(tl.program_id(0), tl.int64), count, tl.cast(tl.num_programs(0), tl.int64)):
        i_k = task % NK
        i_c = task // NK % NC
        i_t = task // (NK * NC) % NT
        i_bh = task // (NK * NC * NT)
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
            i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            length = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
        else:
            bos = i_b * T
            length = T
        start = i_t * BT + i_c * BC
        if start < length:
            o_k = i_k * BK + tl.arange(0, BK)
            base = (bos * H + i_h) * K
            base_a = (bos * H + i_h) * BT
            b_dq, b_dk, b_dkt = gdn2_bwd_diagonal(
                q + base, k + base, b + base, g_raw + base, dAqk + base_a, dAkk + base_a,
                start, length, o_k, H, K, BT, BC,
            )
            o_t = start + tl.arange(0, BC)
            offsets = ((bos + o_t[:, None]) * H + i_h) * K + o_k[None, :]
            mask = (o_t[:, None] < length) & (o_k[None, :] < K)
            tl.store(dq + offsets, b_dq, mask)
            tl.store(dk + offsets, b_dk, mask)
            tl.store(dkt + offsets, b_dkt, mask)


def gdn2_bwd_diagonal_precompute(
    q: torch.Tensor,
    k: torch.Tensor,
    b: torch.Tensor,
    g_raw: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_size: int,
    subchunk_size: int,
    key_tile_size: int,
):
    """Compute FP32 diagonal contributions in an Ascend vector kernel."""
    B, T, H, K = q.shape
    NT = triton.cdiv(T, chunk_size) if chunk_indices is None else len(chunk_indices)
    dq, dk, dkt = [torch.empty_like(q, dtype=torch.float32) for _ in range(3)]
    gdn2_bwd_kernel_diagonal[(get_multiprocessor_count(),)](
        q=q,
        k=k,
        b=b,
        g_raw=g_raw,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        dkt=dkt,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        NT=NT,
        H=H,
        K=K,
        BT=chunk_size,
        BC=subchunk_size,
        BK=key_tile_size,
        IS_VARLEN=cu_seqlens is not None,
    )
    return dq, dk, dkt
