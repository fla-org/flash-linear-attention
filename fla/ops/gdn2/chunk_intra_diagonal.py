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
    # For each destination i, scan only gates j+1 through i. A large gate
    # outside this interval cannot erase a later decay through subtraction.
    interval = tl.where(o_i[:, None, None] > o_i[None, :, None], b_g[None, :, :], 0.)
    decay = exp2(tl.cumsum(interval, axis=1, reverse=True) * 1.4426950408889634)
    valid = (o_i[:, None] >= o_i[None, :]) & (o_t[:, None] < T) & (o_t[None, :] < T)
    offsets_a = o_t[:, None] * H * BT + start % BT + o_i[None, :]
    b_daq = tl.load(dAqk + offsets_a, mask=valid, other=0).to(tl.float32)
    b_dak = tl.load(dAkk + offsets_a, mask=valid, other=0).to(tl.float32)
    dq = tl.sum(b_daq[:, :, None] * b_k[None, :, :] * decay, axis=1)
    dk = tl.sum(b_dak[:, :, None] * b_k[None, :, :] * decay, axis=1)
    dkt = tl.sum(b_daq[:, :, None] * b_q[:, None, :] * decay, axis=0)
    dkt += tl.sum(b_dak[:, :, None] * (b_k * b_b)[:, None, :] * decay, axis=0)
    return dq, dk, dkt
