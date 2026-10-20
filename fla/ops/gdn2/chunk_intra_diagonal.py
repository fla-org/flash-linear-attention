# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl

from fla.ops.utils.op import exp2


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
    # Tile destination rows so scan and reduction temporaries fit Ascend UB.
    for block in range(tl.cdiv(BC, BR)):
        o_r = block * BR + tl.arange(0, BR)
        row_indices = tl.broadcast_to(o_r[:, None], (BR, BK))
        b_qr = tl.gather(b_q, row_indices, axis=0)
        b_kr = tl.gather(b_k, row_indices, axis=0)
        b_br = tl.gather(b_b, row_indices, axis=0)
        # The reverse scan includes only gates j+1 through destination r.
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
