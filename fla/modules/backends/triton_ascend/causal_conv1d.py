# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Causal 1D convolution kernels adapted for triton-ascend on Huawei NPU."""

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.utils import prepare_chunk_indices
from fla.utils import get_multiprocessor_count, input_guard

# triton-ascend exposes extract_slice/insert_slice via cann.extension on some versions.
try:
    from triton.language.extra.cann.extension import extract_slice, insert_slice

    if not hasattr(tl, 'extract_slice'):
        tl.extract_slice = extract_slice
    if not hasattr(tl, 'insert_slice'):
        tl.insert_slice = insert_slice
except ImportError:
    pass


def _select_coregrid_bd(D: int, preferred: int) -> int | None:
    """Largest power-of-2 BD <= preferred that exactly divides D.

    Core-grid kernels need channel tiles on exact D boundaries;
    odd widths (e.g. D=200) fall back to the legacy path.
    """
    bd = min(preferred, triton.next_power_of_2(max(D, 1)))
    while bd > 8 and (bd > D or D % bd != 0):
        bd //= 2
    if bd > D or D % bd != 0:
        return None
    return bd


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'HAS_Y_LINEAR': lambda args: args['y_linear'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_fwd_coregrid_kernel(
    x,
    y,
    y_linear,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_Y_LINEAR: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_CHKS: tl.int32,
    NUM_BLKS_D: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_tasks = NUM_BLKS_D * NUM_CHKS

    for task_id in range(pid, total_tasks, num_programs):
        i_d = task_id % NUM_BLKS_D
        i_chk = task_id // NUM_BLKS_D

        if IS_VARLEN:
            i_n = tl.load(chunk_indices + i_chk * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_chk * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)
            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq
            i_n = i_b
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T
            T_len = T

        t0 = tl.cast(i_t, tl.int64) * BT
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D

        # tail-of-allocation guard: packed-row block end must not exceed B*T,
        # else MTE DMA touches unmapped pages.
        is_tail_chunk = (bos + t0 + BT) > (tl.cast(B, tl.int64) * T)

        if HAS_WEIGHT:
            # host layout is [W, D] (contiguous along D), so the block load is stride-1.
            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w)

        b_y = tl.zeros((BT, BD), dtype=tl.float32)
        yi_offset_1 = o_d[None, :]

        # split the constexpr flag from the runtime `i_t * BT >= W` check.
        # triton-ascend still lowers the else if they are or-ed, and would compile
        # `initial_state + ...` when the pointer is None.
        if not USE_INITIAL_STATE:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = t0 + i_w + tl.arange(0, BT)[:, None]
                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                # intra-loop load: preloading overflows UB under certain tiling.
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi
        elif t0 >= W:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = t0 + i_w + tl.arange(0, BT)[:, None]
                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi
        else:
            o_t = t0 + tl.arange(0, BT)
            for i_w in tl.static_range(-W + 1, 1):
                o_x = o_t + i_w
                m_x = ((o_x >= 0) & (o_x < T_len))[:, None] & m_d
                m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d
                b_yi = tl.load(x + bos * D + o_x[:, None] * D + o_d, mask=m_x, other=0).to(tl.float32)
                b_yi += tl.load(
                    initial_state + tl.cast(i_n, tl.int64) * D * W + o_d * W + (o_x + W)[:, None],
                    mask=m_c,
                    other=0,
                ).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi

        if HAS_BIAS:
            b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

        # pre-silu linear y for bwd: extra MTE3 store, avoids a second full fwd launch.
        if HAS_Y_LINEAR:
            if is_tail_chunk:
                o_t_yl = t0 + tl.arange(0, BT)
                m_t_yl = (o_t_yl >= 0) & (o_t_yl < T_len)
                b_yl_cast = tl.cast(b_y, dtype=y_linear.dtype.element_ty, fp_downcast_rounding='rtne')
                tl.store(
                    y_linear + bos * D + o_t_yl[:, None] * D + o_d[None, :],
                    b_yl_cast,
                    mask=m_t_yl[:, None] & m_d[None, :],
                )
            else:
                p_yl = tl.make_block_ptr(
                    y_linear + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0),
                )
                tl.store(
                    p_yl,
                    tl.cast(b_y, dtype=p_yl.dtype.element_ty, fp_downcast_rounding='rtne'),
                    boundary_check=(0, 1),
                )

        if ACTIVATION == 'swish' or ACTIVATION == 'silu':
            b_y = b_y * tl.sigmoid(b_y)

        if HAS_RESIDUAL:
            if is_tail_chunk:
                o_t_r = t0 + tl.arange(0, BT)
                m_t_r = (o_t_r >= 0) & (o_t_r < T_len)
                b_residual = tl.load(
                    residual + bos * D + o_t_r[:, None] * D + o_d[None, :],
                    mask=m_t_r[:, None] & m_d[None, :],
                    other=0.,
                )
            else:
                p_residual = tl.make_block_ptr(
                    residual + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0),
                )
                b_residual = tl.load(p_residual, boundary_check=(0, 1))
            b_y += b_residual

        if is_tail_chunk:
            o_t_y = t0 + tl.arange(0, BT)
            m_t_y = (o_t_y >= 0) & (o_t_y < T_len)
            b_y_cast = tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding='rtne')
            tl.store(
                y + bos * D + o_t_y[:, None] * D + o_d[None, :],
                b_y_cast,
                mask=m_t_y[:, None] & m_d[None, :],
            )
        else:
            p_y = tl.make_block_ptr(y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            tl.store(
                p_y,
                tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'),
                boundary_check=(0, 1),
            )


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_bwd_coregrid_kernel(
    x,
    y,
    weight,
    initial_state,
    dh0,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_BLKS_D: tl.int32,
    NUM_CHKS: tl.int32,
    NT_STRIDE: tl.int32,
    I_T_OFFSET: tl.int32,
    TAIL_MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # packed [B, T, D] row count; tail DMA past B*T faults MTE ("DDR address out of range").
    # TAIL_MODE: 0 = never (block_ptr bulk), 1 = always (masked), 2 = runtime (varlen / NT==1).
    total_rows = tl.cast(B, tl.int64) * T
    total_tasks = NUM_BLKS_D * NUM_CHKS

    for task_id in range(pid, total_tasks, num_programs):
        i_d = task_id % NUM_BLKS_D
        i_chk = task_id // NUM_BLKS_D

        if IS_VARLEN:
            i_tg = tl.cast(i_chk, tl.int64)
            i_n = tl.load(chunk_indices + i_chk * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_chk * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)
            i_b = i_chk // NT_STRIDE
            i_t = i_chk % NT_STRIDE + I_T_OFFSET
            i_tg = tl.cast(i_b, tl.int64) * NT_per_seq + tl.cast(i_t, tl.int64)
            i_n = i_b
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T
            T_len = T

        t0 = tl.cast(i_t, tl.int64) * BT
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D
        if TAIL_MODE == 0:
            is_tail_chunk = False
        elif TAIL_MODE == 1:
            is_tail_chunk = True
        else:
            is_tail_chunk = (bos + t0 + BT + W - 1) > total_rows

        if HAS_WEIGHT:
            if is_tail_chunk:
                o_t_x = t0 + tl.arange(0, BT)
                m_t_x = (o_t_x >= 0) & (o_t_x < T_len)
                b_x = tl.load(
                    x + bos * D + o_t_x[:, None] * D + o_d[None, :],
                    mask=m_t_x[:, None] & m_d[None, :],
                    other=0,
                )
            else:
                p_x = tl.make_block_ptr(x + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                b_x = tl.load(p_x, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1), padding_option='zero')

        b_dx = tl.zeros((BT, BD), dtype=tl.float32)
        if HAS_BIAS:
            b_db = tl.zeros((BD,), dtype=tl.float32)

        # no-state: one BT+W-1 load + extract_slice; cache paths use per-tap loads.
        if not USE_FINAL_STATE and not USE_INITIAL_STATE:
            b_dw = tl.zeros((W, BD), dtype=tl.float32)

            if is_tail_chunk:
                o_t_full = t0 + tl.arange(0, BT + W - 1)
                m_t_full = (o_t_full >= 0) & (o_t_full < T_len)
                b_dy = tl.load(
                    dy + bos * D + o_t_full[:, None] * D + o_d[None, :],
                    mask=m_t_full[:, None] & m_d[None, :],
                    other=0.,
                ).to(tl.float32)
                if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                    b_y = tl.load(
                        y + bos * D + o_t_full[:, None] * D + o_d[None, :],
                        mask=m_t_full[:, None] & m_d[None, :],
                        other=0.,
                    ).to(tl.float32)
            else:
                p_dy = tl.make_block_ptr(
                    dy + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT + W - 1, BD), (1, 0),
                )
                b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                    p_y = tl.make_block_ptr(
                        y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT + W - 1, BD), (1, 0),
                    )
                    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)

            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

            for i_w in tl.static_range(0, W):
                b_dy_sub = tl.extract_slice(b_dy, [i_w, 0], [BT, BD], [1, 1])
                b_wdy = b_dy_sub
                if HAS_WEIGHT:
                    b_wdy = b_wdy * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])
                    b_dw_sub = tl.sum(b_dy_sub * b_x, 0)
                    b_dw = tl.insert_slice(b_dw, b_dw_sub[None, :], [W - i_w - 1, 0], [1, BD], [1, 1])
                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_sub, 0)
                b_dx += b_wdy

            p_dw = tl.make_block_ptr(dw + i_tg * W * D, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            tl.store(p_dw, b_dw.to(dw.dtype.element_ty))
        elif t0 >= W:
            for i_w in tl.static_range(0, W):
                if is_tail_chunk:
                    o_t_iw = t0 + i_w + tl.arange(0, BT)
                    m_t_iw = (o_t_iw >= 0) & (o_t_iw < T_len)
                    b_dy = tl.load(
                        dy + bos * D + o_t_iw[:, None] * D + o_d[None, :],
                        mask=m_t_iw[:, None] & m_d[None, :],
                        other=0.,
                    ).to(tl.float32)
                    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                        b_y = tl.load(
                            y + bos * D + o_t_iw[:, None] * D + o_d[None, :],
                            mask=m_t_iw[:, None] & m_d[None, :],
                            other=0.,
                        ).to(tl.float32)
                        b_ys = tl.sigmoid(b_y)
                        b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
                else:
                    p_dy = tl.make_block_ptr(
                        dy + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
                    )
                    b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                        p_y = tl.make_block_ptr(
                            y + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
                        )
                        b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                        b_ys = tl.sigmoid(b_y)
                        b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
                b_wdy = b_dy
                if HAS_WEIGHT:
                    b_wdy = b_wdy * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])
                    b_dw = tl.sum(b_dy * b_x, 0)
                    tl.store(dw + i_tg * W * D + (W - i_w - 1) * D + o_d, b_dw.to(dw.dtype.element_ty), mask=m_d)
                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy, 0)
                b_dx += b_wdy
        else:
            o_t = t0 + tl.arange(0, BT)
            for i_w in tl.static_range(0, W):
                if is_tail_chunk:
                    o_t_iw = t0 + i_w + tl.arange(0, BT)
                    m_t_iw = (o_t_iw >= 0) & (o_t_iw < T_len)
                    b_dy_shift = tl.load(
                        dy + bos * D + o_t_iw[:, None] * D + o_d[None, :],
                        mask=m_t_iw[:, None] & m_d[None, :],
                        other=0.,
                    ).to(tl.float32)
                    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                        b_y = tl.load(
                            y + bos * D + o_t_iw[:, None] * D + o_d[None, :],
                            mask=m_t_iw[:, None] & m_d[None, :],
                            other=0.,
                        ).to(tl.float32)
                        b_ys = tl.sigmoid(b_y)
                        b_dy_shift = b_dy_shift * b_ys * (1 + b_y * (1 - b_ys))
                else:
                    p_dy = tl.make_block_ptr(
                        dy + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
                    )
                    b_dy_shift = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                        p_y = tl.make_block_ptr(
                            y + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0),
                        )
                        b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                        b_ys = tl.sigmoid(b_y)
                        b_dy_shift = b_dy_shift * b_ys * (1 + b_y * (1 - b_ys))
                if HAS_WEIGHT:
                    b_dw = tl.sum(b_dy_shift * b_x, 0)
                    if USE_INITIAL_STATE:
                        mask_head_rows = o_t < i_w
                        b_dy_head = tl.load(
                            dy + bos * D + o_t[:, None] * D + o_d,
                            mask=(mask_head_rows[:, None] & m_d[None, :]),
                            other=0.,
                        ).to(tl.float32)
                        if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                            b_y_head = tl.load(
                                y + bos * D + o_t[:, None] * D + o_d,
                                mask=(mask_head_rows[:, None] & m_d[None, :]),
                                other=0.,
                            ).to(tl.float32)
                            b_ys_head = tl.sigmoid(b_y_head)
                            b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                        o_c = W - i_w + o_t
                        mask_c = mask_head_rows & (o_c >= 1) & (o_c < W)
                        b_xc = tl.load(
                            initial_state + tl.cast(i_n, tl.int64) * D * W + o_d[None, :] * W + o_c[:, None],
                            mask=(mask_c[:, None] & m_d[None, :]),
                            other=0.,
                        ).to(tl.float32)
                        b_dw += tl.sum(b_dy_head * b_xc, 0)
                    tl.store(dw + i_tg * W * D + (W - i_w - 1) * D + o_d, b_dw.to(dw.dtype.element_ty), mask=m_d)

                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_shift, 0)
                if HAS_WEIGHT:
                    b_wdy = b_dy_shift * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])
                else:
                    b_wdy = b_dy_shift
                b_dx += b_wdy

            if USE_INITIAL_STATE:
                for i_w in tl.static_range(1, W):
                    # dh0[i_w] = sum_{t=0}^{i_w-1} dy0[t] * w[i_w-1-t];
                    # load dy0 row-by-row to avoid overflowing UB with a [BT, BD] preload.
                    b_dh0_s = tl.zeros((BD,), dtype=tl.float32)
                    for i_t2 in tl.static_range(0, W - 1):
                        if i_t2 < i_w:
                            dy0_row = tl.load(
                                dy + bos * D + (t0 + i_t2) * D + o_d, mask=m_d, other=0.,
                            ).to(tl.float32)
                            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                                y0_row = tl.load(
                                    y + bos * D + (t0 + i_t2) * D + o_d, mask=m_d, other=0.,
                                ).to(tl.float32)
                                y0_s = tl.sigmoid(y0_row)
                                dy0_row = dy0_row * y0_s * (1 + y0_row * (1 - y0_s))
                            if HAS_WEIGHT:
                                w_row = tl.extract_slice(b_w, [i_w - 1 - i_t2, 0], [1, BD], [1, 1])
                                b_dh0_s += tl.sum(dy0_row[None, :] * w_row, 0).to(tl.float32)
                            else:
                                b_dh0_s += dy0_row
                    tl.store(
                        dh0 + tl.cast(i_t, tl.int64) * tl.cast(B, tl.int64) * D * W +
                        tl.cast(i_n, tl.int64) * D * W + o_d * W + i_w,
                        b_dh0_s.to(dh0.dtype.element_ty, fp_downcast_rounding='rtne'),
                        mask=m_d,
                    )

        if HAS_BIAS:
            b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne')
            tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

        if USE_FINAL_STATE:
            if t0 + BT >= T_len - W:
                # final_state[b, d, w] = x[b, T_len-W+w, d] for w in [0, W),
                # so dx[t] += dht[b, d, t-(T_len-W)] for t in [T_len-W, T_len).
                row_arange = tl.arange(0, BT)
                for i_w in tl.static_range(0, W):
                    target_row = T_len - W + i_w
                    local_row = target_row - t0
                    in_chunk = (local_row >= 0) & (local_row < BT) & (target_row >= 0) & (target_row < T_len)
                    b_dht_row = tl.load(dht + tl.cast(i_n, tl.int64) * D * W + o_d *
                                        W + i_w, mask=m_d, other=0.).to(tl.float32)
                    row_match = (row_arange == local_row) & in_chunk
                    b_dx += tl.where(row_match[:, None] & m_d[None, :], b_dht_row[None, :], 0.)

        if is_tail_chunk:
            o_t_dx = t0 + tl.arange(0, BT)
            m_t_dx = (o_t_dx >= 0) & (o_t_dx < T_len)
            b_dx_cast = tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding='rtne')
            tl.store(
                dx + bos * D + o_t_dx[:, None] * D + o_d[None, :],
                b_dx_cast,
                mask=m_t_dx[:, None] & m_d[None, :],
            )
        else:
            p_dx = tl.make_block_ptr(dx + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            tl.store(
                p_dx,
                tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding='rtne'),
                boundary_check=(0, 1),
            )


# Ascend Triton rejects grids whose product exceeds 65535 (see fla/modules/token_shift.py).
_NPU_MAX_TRITON_GRID = 65535
_ELEM_BLOCK = 2048


def _elementwise_launch_iters(numel: int):
    n_blocks = triton.cdiv(numel, _ELEM_BLOCK)
    for block_off in range(0, n_blocks, _NPU_MAX_TRITON_GRID):
        yield min(_NPU_MAX_TRITON_GRID, n_blocks - block_off), block_off * _ELEM_BLOCK


def _npu_chunk_size(T: int, BT: int) -> int:
    BT = min(max(BT, 1), 64)
    if BT not in (1, 2, 4, 8, 16, 32, 64):
        BT = triton.next_power_of_2(BT)
    # Ascend compiler requires power-of-2 BT; pad with mask when BT > T.
    if T not in (1, 2, 4, 8, 16, 32, 64):
        BT = min(triton.next_power_of_2(T), 64)
    else:
        BT = min(BT, T, 64)
    return BT


def _clamp_bd_for_grid(B: int, NT: int, D: int, BD: int) -> int:
    while triton.cdiv(D, BD) * NT * B > _NPU_MAX_TRITON_GRID and BD < 64:
        BD *= 2
    return BD


def _npu_max_axis_chunks(grid_dim0: int, batch: int = 1) -> int:
    denom = grid_dim0 * batch
    if denom > _NPU_MAX_TRITON_GRID:
        raise RuntimeError(
            f'Ascend Triton grid dim0*batch={denom} exceeds {_NPU_MAX_TRITON_GRID}',
        )
    return max(1, _NPU_MAX_TRITON_GRID // max(denom, 1))


# bf16/fp16 use big tiles (BT*BD <= 16384, swept on 910B; ~19x on large-T forward).
# fp32 keeps the original conservative tiles (4B/elem overflows Ascend UB at big tiles).
_NPU_FWD_TILE_BUDGET = 16384
_NPU_FWD_MAX_BT = 512


def _npu_tile_config(
    T: int,
    BT: int,
    D: int,
    dtype: torch.dtype,
    initial_state: torch.Tensor | None,
) -> tuple[int, int]:
    if dtype not in (torch.bfloat16, torch.float16):
        BT = _npu_chunk_size(T, BT)
        BD = 16
        if D >= 8192:
            BD = 8
            BT = min(BT, 8)
        elif D >= 1024:
            # BD=4 overflows Ascend UB on large-D forward; cap BT to limit NT.
            BD = 8
            BT = min(BT, 32)
        elif D >= 512:
            BD = 8
        if dtype == torch.float16 and initial_state is not None:
            BD = min(BD, 8)
        if dtype == torch.bfloat16 and T <= 16:
            BD = 8
        return BD, BT

    pow2_t = triton.next_power_of_2(T)
    floor_t = pow2_t if pow2_t == T else pow2_t // 2      # largest power-of-2 <= T
    BT = min(_NPU_FWD_MAX_BT, floor_t)
    budget = _NPU_FWD_TILE_BUDGET
    if D >= 8192:
        budget = min(budget, 8192)
    BD = max(8, triton.next_power_of_2(budget // BT))
    BD = min(BD, 64)
    if dtype == torch.float16 and initial_state is not None:
        BD = min(BD, 8)
    if dtype == torch.bfloat16 and T <= 16:
        BD = min(BD, 8)
    return BD, BT


def _npu_bwd_tile_config(
    T: int,
    BT: int,
    D: int,
    dtype: torch.dtype,
    initial_state: torch.Tensor | None,
) -> tuple[int, int]:
    BT = _npu_chunk_size(T, BT)
    BD = 16
    if initial_state is not None:
        BD = min(BD, 8)
        BT = min(BT, 32)
    if D >= 2048:
        BD = 8
        BT = min(BT, 8)
    elif D >= 1024:
        BD = 8
        BT = min(BT, 16)
    elif D >= 512:
        BD = 8
        BT = min(BT, 32)
    if dtype == torch.bfloat16 and T <= 16:
        BD = 8
        BT = 32
    return BD, BT


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NT: tl.constexpr,
    NT_GRID: tl.constexpr,
    MAX_NT_PER_BLOCK: tl.constexpr,
    NT_GRID_OFFSET: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Ported from ant_AReaL: block_ptr (coalesced loads) + fused bias/silu/residual,
    # with MAX_NT_PER_BLOCK T-chunks per program to shrink the grid. The host launches the
    # NT_GRID axis in chunks (NT_GRID_OFFSET) so the grid product stays under the 65535
    # Ascend cap. Masked scalar loads are kept only for the initial_state head edge.
    i_d, i_t_base, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t_base = i_t_base + NT_GRID_OFFSET

    if IS_VARLEN:
        i_n_base = tl.load(chunk_indices + i_t_base * MAX_NT_PER_BLOCK * 2).to(tl.int32)
    else:
        i_n_base = i_b
        bos_base = tl.cast(i_b, tl.int64) * T
        eos_base = bos_base + T

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW)
    m_d = o_d < D
    m_w = o_w < W

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    for i_t_iter in tl.static_range(0, MAX_NT_PER_BLOCK):
        i_t_global = i_t_base * MAX_NT_PER_BLOCK + i_t_iter
        if i_t_global < NT:
            if IS_VARLEN:
                i_n, i_t = tl.load(chunk_indices + i_t_global * 2).to(tl.int32), tl.load(chunk_indices +
                                                                                         i_t_global * 2 + 1).to(tl.int32)
                bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
                T_local = (eos - bos).to(tl.int32)
                p_x = x + bos * stride_x_t
            else:
                i_n = i_n_base
                i_t = i_t_global
                bos, eos = bos_base, eos_base
                T_local = T
                p_x = x + tl.cast(i_b, tl.int64) * stride_x_n

            t0 = tl.cast(i_t, tl.int64) * BT
            b_y = tl.zeros((BT, BD), dtype=tl.float32)
            if not USE_INITIAL_STATE or t0 >= W:
                for i_w in tl.static_range(W):
                    p_yi = tl.make_block_ptr(p_x, (T_local, D), (stride_x_t, stride_x_d),
                                             (i_t * BT + i_w - W + 1, i_d * BD), (BT, BD), (1, 0))
                    b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
                    if HAS_WEIGHT:
                        b_yi *= tl.sum(b_w * (o_w == i_w), 1)
                    b_y += b_yi
            else:
                o_t = t0 + tl.arange(0, BT)
                for i_w in tl.static_range(W):
                    o_x = o_t + i_w - W + 1
                    # Explicit 2D ([None, :]) indexing throughout: triton-ascend miscompiles
                    # the implicit 1D-vs-2D broadcast (o_d / m_d against o_x[:, None]) in
                    # these scalar pointer loads, faulting the vector core at runtime.
                    m_x = ((o_x >= 0) & (o_x < T_local))[:, None] & m_d[None, :]
                    m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d[None, :]
                    b_yi = tl.load(
                        p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
                        mask=m_x,
                        other=0,
                    ).to(tl.float32)
                    # Guard with a pure-constexpr check: triton-ascend does not fold the
                    # outer `not USE_INITIAL_STATE or <runtime>`, so this else branch is
                    # lowered even when initial_state is None -- without the guard it would
                    # dereference None at compile time (AttributeError on None.type).
                    if USE_INITIAL_STATE:
                        b_yi += tl.load(initial_state + tl.cast(i_n, tl.int64) * D * W + o_d[None, :] * W + (o_x + W)
                                        [:, None], mask=m_c, other=0).to(tl.float32)
                    if HAS_WEIGHT:
                        b_yi *= tl.sum(b_w * (o_w == i_w), 1)[None, :]
                    b_y += b_yi

            if HAS_BIAS:
                b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                b_y = b_y * tl.sigmoid(b_y)
            if HAS_RESIDUAL:
                p_residual = tl.make_block_ptr(residual + bos * D, (T_local, D), (D, 1),
                                               (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                b_y += tl.load(p_residual, boundary_check=(0, 1))

            p_y = tl.make_block_ptr(y + bos * D, (T_local, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.jit
def _silu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    ELEM_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK) + ELEM_OFFSET
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.).to(tl.float32)
    y = x * tl.sigmoid(x)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    ELEM_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK) + ELEM_OFFSET
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.).to(tl.float32)
    tl.store(out_ptr + offs, (a + b).to(out_ptr.dtype.element_ty), mask=mask)


def _launch_silu(y: torch.Tensor) -> torch.Tensor:
    y = y.contiguous()
    out = torch.zeros_like(y)
    n = y.numel()
    for grid, elem_off in _elementwise_launch_iters(n):
        _silu_kernel[(grid,)](
            y, out, n,
            ELEM_OFFSET=elem_off,
            BLOCK=_ELEM_BLOCK,
        )
    return out


def _launch_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    b = b.contiguous()
    out = torch.zeros_like(a)
    n = a.numel()
    for grid, elem_off in _elementwise_launch_iters(n):
        _add_kernel[(grid,)](
            a, b, out, n,
            ELEM_OFFSET=elem_off,
            BLOCK=_ELEM_BLOCK,
        )
    return out


@triton.jit
def _silu_bwd_kernel(
    y_ptr,
    dy_ptr,
    out_ptr,
    n_elements,
    ELEM_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Flat contiguous elementwise silu/sigmoid backward. Inputs are [B,T,D] contiguous, so
    # the element offset IS the tensor offset -- no per-element modulo / strided gather
    # (the old form was the #1 hotspot: 71% of fwd+bwd on PipeUtilization, dominated by
    # 4 integer divisions per element on the vector pipe).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK) + ELEM_OFFSET
    mask = offs < n_elements
    y = tl.load(y_ptr + offs, mask=mask, other=0.).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.).to(tl.float32)
    s = tl.sigmoid(y)
    out = dy * s * (1.0 + y * (1.0 - s))
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_silu_bwd(y_pre: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    # Flat indexing requires contiguous inputs; conv output (y_pre) is contiguous, dy may
    # arrive non-contiguous from upstream -- make it contiguous (no-op copy if already so).
    if not y_pre.is_contiguous():
        y_pre = y_pre.contiguous()
    if not dy.is_contiguous():
        dy = dy.contiguous()
    out = torch.zeros_like(dy, memory_format=torch.contiguous_format)
    n = dy.numel()
    for grid, elem_off in _elementwise_launch_iters(n):
        _silu_bwd_kernel[(grid,)](
            y_pre, dy, out, n,
            ELEM_OFFSET=elem_off,
            BLOCK=_ELEM_BLOCK,
        )
    return out


def _postprocess_fwd(
    y: torch.Tensor,
    residual: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    if activation in ('swish', 'silu'):
        y = _launch_silu(y)
    if residual is not None:
        if residual.stride() != y.stride():
            residual = residual.contiguous()
        y = _launch_add(y, residual)
    return y


def _use_seq_bwd(
    T: int,
    dtype: torch.dtype,
    initial_state: torch.Tensor | None,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    return (
        cu_seqlens is None
        and initial_state is None
        and dht is None
        and dtype == torch.bfloat16
        and T <= 16
    )


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
})
@triton.jit
def causal_conv1d_bwd_seq_kernel(
    x,
    weight,
    dy,
    dx,
    dw,
    db,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    B,
    TC: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_elements = B * TC * D
    mask = offs < n_elements
    d = offs % D
    tmp = offs // D
    t = tmp % TC
    b = tmp // TC

    b_dx = tl.zeros((BLOCK,), dtype=tl.float32)
    for i_w in tl.static_range(0, W):
        t_dy = t + i_w
        dy_off = b * stride_dy_n + t_dy * stride_dy_t + d * stride_dy_d
        b_dy = tl.load(dy + dy_off, mask=mask & (t_dy < TC), other=0.).to(tl.float32)
        if HAS_WEIGHT:
            w_idx = W - i_w - 1
            b_w = tl.load(weight + d * W + w_idx, mask=mask, other=0.).to(tl.float32)
            b_dx += b_dy * b_w
        else:
            b_dx += b_dy

    dx_off = b * stride_dx_n + t * stride_dx_t + d * stride_dx_d
    tl.store(dx + dx_off, b_dx.to(dx.dtype.element_ty), mask=mask)

    if HAS_WEIGHT:
        x_off = b * stride_x_n + t * stride_x_t + d * stride_x_d
        b_x = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
        i_tg = b * TC + t
        for i_w in tl.static_range(0, W):
            t_dy = t + i_w
            dy_off = b * stride_dy_n + t_dy * stride_dy_t + d * stride_dy_d
            b_dy = tl.load(dy + dy_off, mask=mask & (t_dy < TC), other=0.).to(tl.float32)
            w_idx = W - i_w - 1
            tl.store(
                dw + (i_tg * D + d) * W + w_idx,
                (b_dy * b_x).to(dw.dtype.element_ty),
                mask=mask,
            )

    if HAS_BIAS:
        i_tg = b * TC + t
        dy_off = b * stride_dy_n + t * stride_dy_t + d * stride_dy_d
        b_dy0 = tl.load(dy + dy_off, mask=mask, other=0.)
        tl.store(db + i_tg * D + d, b_dy0.to(db.dtype.element_ty), mask=mask)


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_bwd_dx_kernel(
    dy,
    weight,
    dht,
    dx,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
):
    # dx-only backward: needs only dy and weight, so it compiles at fwd-sized tiles.
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        p_dy = dy + bos * stride_dy_t
        p_dx = dx + bos * stride_dx_t
    else:
        i_n = i_b
        i_t = i_t + CHUNK_OFFSET
        p_dy = dy + tl.cast(i_b, tl.int64) * stride_dy_n
        p_dx = dx + tl.cast(i_b, tl.int64) * stride_dx_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
    m_t = (o_t >= 0) & (o_t < T)

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in tl.static_range(0, W):
        o_dy = o_t + i_w
        m_dy = ((o_dy >= 0) & (o_dy < T))[:, None] & m_d[None, :]
        b_dy = tl.load(
            p_dy + o_dy[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d,
            mask=m_dy,
            other=0,
        ).to(tl.float32)
        if HAS_WEIGHT:
            b_dx += b_dy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)[None, :]
        else:
            b_dx += b_dy

    if USE_FINAL_STATE:
        if tl.cast(i_t, tl.int64) * BT + BT >= T - W:
            start_tok = T - (W - 1)
            offset = o_t
            tok_idx = offset - start_tok
            mask = (offset >= start_tok) & (offset < T)
            w_idx = 1 + tok_idx
            dht_off = tl.cast(i_n, tl.int64) * D * W + o_d[None, :] * W + w_idx[:, None]
            b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.).to(tl.float32)
            b_dx += b_dht

    tl.store(
        p_dx + o_t[:, None] * stride_dx_t + o_d[None, :] * stride_dx_d,
        tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_t[:, None] & m_d[None, :],
    )


def _launch_bwd_dx_core(
    dy,
    weight,
    dht,
    cu_seqlens,
    cu_seqlens_cpu,
    B,
    T,
    D,
    W,
    BT,
    BD=None,
):
    # dx-only kernel: fwd big tiles, except dht (USE_FINAL_STATE) falls back to small
    # tiles (the extra branch overflows Ascend UB at big tiles).
    if BD is None:
        if dht is not None:
            pow2_t = triton.next_power_of_2(T)
            floor_t = pow2_t if pow2_t == T else pow2_t // 2
            BT = min(64, floor_t)
            BD = min(16, max(8, triton.next_power_of_2(D)))
        else:
            BD, BT = _npu_tile_config(T, BT, D, dy.dtype, None)
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, BT)
    BD = _clamp_bd_for_grid(B, NT, D, BD)
    BW = triton.next_power_of_2(W)

    stride_dy_n, stride_dy_t, stride_dy_d = dy.stride()
    dx = torch.zeros_like(dy, memory_format=torch.contiguous_format)
    stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()

    max_nt = _npu_max_axis_chunks(triton.cdiv(D, BD), B)
    kernel_kwargs = dict(
        dy=dy,
        weight=weight,
        dht=dht,
        dx=dx,
        cu_seqlens=cu_seqlens,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        BD=BD,
        stride_dy_n=stride_dy_n,
        stride_dy_t=stride_dy_t,
        stride_dy_d=stride_dy_d,
        stride_dx_n=stride_dx_n,
        stride_dx_t=stride_dx_t,
        stride_dx_d=stride_dx_d,
    )
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        grid = (triton.cdiv(D, BD), nt_len, B)
        if cu_seqlens is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['CHUNK_OFFSET'] = 0
        else:
            kernel_kwargs['chunk_indices'] = None
            kernel_kwargs['CHUNK_OFFSET'] = nt_off
        causal_conv1d_bwd_dx_kernel[grid](**kernel_kwargs)
    return dx


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_bwd_dwdb_kernel(
    x,
    dy,
    initial_state,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
    NT: tl.constexpr,
):
    # dw/db-only backward: weight/bias gradient partials (dx is handled separately).
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = tl.cast(i_t, tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        p_x = x + bos * stride_x_t
        p_dy = dy + bos * stride_dy_t
    else:
        i_t = i_t + CHUNK_OFFSET
        i_tg = tl.cast(i_b, tl.int64) * NT + tl.cast(i_t, tl.int64)
        i_n = i_b
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n
        p_dy = dy + tl.cast(i_b, tl.int64) * stride_dy_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
    m_d = o_d < D
    m_t = (o_t >= 0) & (o_t < T)

    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    if HAS_WEIGHT:
        b_x = tl.load(
            p_x + o_t[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=m_t[:, None] & m_d[None, :],
            other=0,
        ).to(tl.float32)

    for i_w in tl.static_range(0, W):
        o_dy = o_t + i_w
        m_dy = ((o_dy >= 0) & (o_dy < T))[:, None] & m_d[None, :]
        b_dy = tl.load(
            p_dy + o_dy[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d,
            mask=m_dy,
            other=0,
        ).to(tl.float32)

        if HAS_WEIGHT:
            b_dw = tl.sum(b_dy * b_x, 0)
            if USE_INITIAL_STATE:
                mask_head_rows = (o_t < i_w) & (o_t < T)
                b_dy_head = tl.load(
                    p_dy + o_t[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d,
                    mask=(mask_head_rows[:, None] & m_d[None, :]),
                    other=0.0,
                ).to(tl.float32)
                o_c = W - i_w + o_t
                mask_c = (mask_head_rows & (o_c >= 1) & (o_c < W))
                b_xc = tl.load(
                    initial_state + tl.cast(i_n, tl.int64) * D * W + o_d[None, :] * W + o_c[:, None],
                    mask=(mask_c[:, None] & m_d[None, :]),
                    other=0.0,
                ).to(tl.float32)
                b_dw += tl.sum(b_dy_head * b_xc, 0)
            tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

        if HAS_BIAS and i_w == 0:
            b_db += tl.sum(b_dy, 0)

    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne')
        tl.store(db + i_tg * D + o_d, b_db, mask=m_d)


def _launch_bwd_dwdb_core(
    x: torch.Tensor,
    dy: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    cu_seqlens_cpu: torch.LongTensor | None,
    B: int,
    T: int,
    D: int,
    W: int,
    BT: int,
    BD: int | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
):
    # dw/db-only kernel (reads x AND dy, so it hits the MLIR/multi-buffer UB cliff before
    # dx-only). Empirically the tile is capped at BT*BD<=1024 (BT=64, BD<=16): profiling
    # showed it Vector-bound (aiv_vec_ratio ~0.91), but BOTH raising BT->128 and BD->32
    # overflow Ascend UB at compile time (the static_range(W) loop + multi-buffer keep a
    # large fp32 live set). So tile tuning is a dead end here; the launch loop handles
    # large grids. Don't _clamp_bd_for_grid (raising BD crashes it).
    if BD is None:
        pow2_t = triton.next_power_of_2(T)
        floor_t = pow2_t if pow2_t == T else pow2_t // 2
        BT = min(64, floor_t)
        BD = min(16, max(8, triton.next_power_of_2(D)))
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, BT)
    BW = triton.next_power_of_2(W)

    stride_x_n, stride_x_t, stride_x_d = x.stride()
    stride_dy_n, stride_dy_t, stride_dy_d = dy.stride()
    dw = weight.new_empty(B * NT, *weight.shape, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B * NT, *bias.shape, dtype=torch.float) if bias is not None else None

    max_nt = _npu_max_axis_chunks(triton.cdiv(D, BD), B)
    kernel_kwargs = dict(
        x=x,
        dy=dy,
        initial_state=initial_state,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        BD=BD,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_dy_n=stride_dy_n,
        stride_dy_t=stride_dy_t,
        stride_dy_d=stride_dy_d,
        NT=NT,
    )
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        grid = (triton.cdiv(D, BD), nt_len, B)
        if cu_seqlens is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['CHUNK_OFFSET'] = 0
            kernel_kwargs['dw'] = dw[nt_off:nt_off + nt_len] if weight is not None else None
            kernel_kwargs['db'] = db[nt_off:nt_off + nt_len] if bias is not None else None
        else:
            kernel_kwargs['chunk_indices'] = None
            kernel_kwargs['CHUNK_OFFSET'] = nt_off
            kernel_kwargs['dw'] = dw
            kernel_kwargs['db'] = db
        causal_conv1d_bwd_dwdb_kernel[grid](**kernel_kwargs)
    return dw, db


@triton.heuristics({
    'USE_ACTIVATION': lambda args: args['y'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def compute_dh0_kernel(
    dy,
    y,
    weight,
    dh0,
    cu_seqlens,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    stride_y_n,
    stride_y_t,
    stride_y_d,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.cast(tl.program_id(1) + CHUNK_OFFSET, tl.int64)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
        dy_base = dy + bos * stride_dy_t
    else:
        seq_len = T
        dy_base = dy + i_n * stride_dy_n

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for i_w in tl.static_range(1, W):
        b_dh0 = tl.zeros([BD], dtype=tl.float32)

        for t in tl.static_range(0, W - 1):
            if t < i_w:
                w_idx = i_w - 1 - t
                p_dy = dy_base + t * stride_dy_t + o_d * stride_dy_d
                m_t = (t < seq_len) & m_d
                b_dy = tl.load(p_dy, mask=m_t, other=0).to(tl.float32)

                if USE_ACTIVATION:
                    if IS_VARLEN:
                        p_y = y + bos * stride_y_t + t * stride_y_t + o_d * stride_y_d
                    else:
                        p_y = y + i_n * stride_y_n + t * stride_y_t + o_d * stride_y_d
                    b_y = tl.load(p_y, mask=m_t, other=0).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

                b_w_col = tl.load(weight + o_d * W + w_idx, mask=m_d, other=0).to(tl.float32)
                b_dh0 += tl.where(m_t, b_dy * b_w_col, 0)

        p_dh0 = dh0 + i_n * D * W + o_d * W + i_w
        tl.store(p_dh0, b_dh0.to(dh0.dtype.element_ty), mask=m_d)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.cast(tl.program_id(1) + CHUNK_OFFSET, tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = (eos - bos).to(tl.int32)
        p_x = x + bos * stride_x_t
    else:
        seq_len = T
        p_x = x + i_n * stride_x_n

    o_w = W - BW + tl.arange(0, BW)
    m_w = o_w >= 0
    o_t = tl.cast(seq_len, tl.int64) - BW + tl.arange(0, BW)
    m_t = (o_t >= 0) & (o_t < seq_len)

    b_x = tl.load(
        p_x + o_t[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
        mask=m_t[:, None] & m_d[None, :],
        other=0,
    ).to(tl.float32)

    if USE_INITIAL_STATE:
        if seq_len < BW:
            o_c = W - (BW - seq_len) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)
            b_cache = tl.load(
                initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                mask=m_d[None, :] & m_c[:, None],
                other=0,
            ).to(tl.float32)
            b_x += b_cache

    p_final = final_state + i_n * D * W + o_d[:, None] * W + o_w[None, :]
    tl.store(p_final, tl.trans(b_x).to(final_state.dtype.element_ty), mask=m_d[:, None] & m_w[None, :])


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
})
@triton.jit
def causal_conv1d_update_kernel(
    x,
    cache,
    y,
    weight,
    bias,
    stride_x_n,
    stride_x_d,
    stride_y_n,
    stride_y_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.cast(tl.program_id(1) + CHUNK_OFFSET, tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    b_x = tl.load(x + i_n * stride_x_n + o_d * stride_x_d, mask=m_d, other=0).to(tl.float32)

    b_y = tl.zeros((BD,), dtype=tl.float32)
    for iw in tl.static_range(0, W):
        if iw < W - 1:
            b_c = tl.load(cache + i_n * D * W + o_d * W + (iw + 1), mask=m_d, other=0).to(tl.float32)
        else:
            b_c = b_x
        tl.store(
            cache + i_n * D * W + o_d * W + iw,
            tl.cast(b_c, dtype=cache.dtype.element_ty, fp_downcast_rounding='rtne'),
            mask=m_d,
        )
        if HAS_WEIGHT:
            b_y += b_c * tl.load(weight + o_d * W + iw, mask=m_d, other=0).to(tl.float32)
        else:
            b_y += b_c

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d)

    tl.store(
        y + i_n * stride_y_n + o_d * stride_y_d,
        tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_d,
    )


def _postprocess_update(
    y: torch.Tensor,
    residual: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    if activation in ('swish', 'silu'):
        y = _launch_silu(y)
    if residual is not None:
        if residual.stride() != y.stride():
            residual = residual.contiguous()
        y = _launch_add(y, residual)
    return y


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_fwd_kernel_scalar(
    x,
    y,
    weight,
    bias,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_y_n,
    stride_y_t,
    stride_y_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
):
    # Scalar (masked-load) forward -- the proven triton-ascend path. Used when
    # initial_state is present: the ant_AReaL block_ptr kernel's initial_state head-edge
    # branch faults the Ascend vector core on this triton-ascend version, so we fall back
    # to this simpler control flow which compiles+runs cleanly.
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        p_x = x + bos * stride_x_t
        p_y = y + bos * stride_y_t
    else:
        i_n = i_b
        i_t = i_t + CHUNK_OFFSET
        bos = tl.cast(i_b, tl.int64) * T
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n
        p_y = y + tl.cast(i_b, tl.int64) * stride_y_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
    m_t = (o_t >= 0) & (o_t < T)
    b_y = tl.zeros((BT, BD), dtype=tl.float32)

    for i_w in tl.static_range(-W + 1, 1):
        o_x = o_t + i_w
        m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :]
        b_yi = tl.load(
            p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=m_x,
            other=0,
        ).to(tl.float32)

        if USE_INITIAL_STATE:
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d[None, :]
            b_yi += tl.load(
                initial_state + tl.cast(i_n, tl.int64) * D * W + o_d[None, :] * W + (o_x + W)[:, None],
                mask=m_c,
                other=0,
            ).to(tl.float32)

        if HAS_WEIGHT:
            b_yi = b_yi * tl.sum(b_w * (o_w == (i_w + W - 1)), 1)[None, :]
        b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)[None, :]

    tl.store(
        p_y + o_t[:, None] * stride_y_t + o_d[None, :] * stride_y_d,
        tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_t[:, None] & m_d[None, :],
    )


def _launch_fwd_core_scalar(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    cu_seqlens_cpu: torch.LongTensor | None,
    B: int,
    T: int,
    D: int,
    W: int,
    BT: int,
) -> torch.Tensor:
    # Scalar-kernel forward (conv only -- no fused activation/residual). Caller applies
    # activation/residual via _postprocess_fwd. Used for the initial_state path.
    # Force conservative (small) tiles regardless of dtype: this is the correctness
    # fallback, and the bf16 big tiles from _npu_tile_config overflow Ascend UB under the
    # scalar masked-load + multi-buffer pattern. Small tiles always fit; perf is secondary
    # here (initial_state is the uncommon path; the fast ant_AReaL kernel handles the rest).
    BD, BT = _npu_tile_config(T, BT, D, torch.float32, initial_state)
    # Rebuild chunk_indices for the (possibly reduced) BT: the caller built them for a
    # different BT, and a BT/chunk_indices mismatch corrupts the varlen (n, t) lookup.
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    else:
        chunk_indices = None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    BD = _clamp_bd_for_grid(B, NT, D, BD)
    BW = triton.next_power_of_2(W)

    stride_x_n, stride_x_t, stride_x_d = x.stride()
    y = torch.zeros_like(x, memory_format=torch.contiguous_format)
    stride_y_n, stride_y_t, stride_y_d = y.stride()

    max_nt = _npu_max_axis_chunks(triton.cdiv(D, BD), B)
    kernel_kwargs = dict(
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        BD=BD,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_t=stride_y_t,
        stride_y_d=stride_y_d,
    )
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        grid = (triton.cdiv(D, BD), nt_len, B)
        if cu_seqlens is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['CHUNK_OFFSET'] = 0
        else:
            kernel_kwargs['chunk_indices'] = chunk_indices
            kernel_kwargs['CHUNK_OFFSET'] = nt_off
        causal_conv1d_fwd_kernel_scalar[grid](**kernel_kwargs)
    return y


def _launch_fwd_core(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    B: int,
    T: int,
    D: int,
    W: int,
    BT: int,
    activation: str | None = None,
) -> torch.Tensor:
    # Ported from ant_AReaL: BD=128 block_ptr + fused bias/silu/residual kernel, with
    # MAX_NT_PER_BLOCK T-chunks per program to shrink the grid (bf16/fp16). float32 falls
    # back to a small tile so the fused kernel still fits the 192KB Ascend UB. The NT_GRID
    # axis is launched in chunks (NT_GRID_OFFSET) so the grid product cdiv(D,BD) x
    # nt_grid_len x B stays under the 65535 Ascend cap.
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    BW = triton.next_power_of_2(W)
    if x.dtype in (torch.bfloat16, torch.float16):
        BD = 128
        MAX_NT_PER_BLOCK = 4
    else:
        # float32: 4B/elem + the fused residual/silu overflows Ascend UB at BD=128.
        BD = 16
        MAX_NT_PER_BLOCK = 1
    NT_GRID = triton.cdiv(NT, MAX_NT_PER_BLOCK)

    stride_x_n, stride_x_t, stride_x_d = x.stride()
    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    grid_dim0 = triton.cdiv(D, BD)
    max_nt_grid = _npu_max_axis_chunks(grid_dim0, B)
    for nt_grid_off in range(0, NT_GRID, max_nt_grid):
        nt_grid_len = min(max_nt_grid, NT_GRID - nt_grid_off)
        grid = (grid_dim0, nt_grid_len, B)
        causal_conv1d_fwd_kernel[grid](
            x=x,
            y=y,
            weight=weight,
            bias=bias,
            residual=residual,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            chunk_indices=chunk_indices,
            B=B,
            T=T,
            D=D,
            W=W,
            BT=BT,
            BW=BW,
            BD=BD,
            NT=NT,
            NT_GRID=NT_GRID,
            MAX_NT_PER_BLOCK=MAX_NT_PER_BLOCK,
            NT_GRID_OFFSET=nt_grid_off,
            ACTIVATION=activation,
            stride_x_n=stride_x_n,
            stride_x_t=stride_x_t,
            stride_x_d=stride_x_d,
        )
    return y


def _launch_fwd_coregrid(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    save_linear: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """1D core-grid forward. Public weight is [D, W]; load as contiguous [W, D].

    If ``save_linear``, also store pre-silu conv+bias (no residual) for bwd.
    """
    B, T, D = x.shape
    W = weight.shape[1]
    weight_wd = weight.transpose(0, 1).contiguous()
    NUM_CORES = get_multiprocessor_count(x.device.index)
    BD = _select_coregrid_bd(D, 256)
    if BD is None:
        raise RuntimeError(f'coregrid fwd: no aligned BD for D={D}')
    BT = min(32, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))
    NUM_BLKS_D = triton.cdiv(D, BD)
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
        NUM_CHKS = len(chunk_indices)
    else:
        chunk_indices = None
        NUM_CHKS = triton.cdiv(T, BT) * B
    y = torch.empty_like(x)
    y_linear = torch.empty_like(x) if save_linear else None
    causal_conv1d_fwd_coregrid_kernel[(NUM_CORES,)](
        x=x,
        y=y,
        y_linear=y_linear,
        weight=weight_wd,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=None,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BD=BD,
        ACTIVATION=activation,
        NUM_CHKS=NUM_CHKS,
        NUM_BLKS_D=NUM_BLKS_D,
    )
    return y, y_linear


def _launch_bwd_coregrid(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    y_linear: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Unified 1D core-grid backward. Public weight is [D, W]; load as contiguous [W, D]."""
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None
    weight_wd = weight.transpose(0, 1).contiguous() if weight is not None else None
    NUM_CORES = get_multiprocessor_count(x.device.index)
    # BT=64 halves dw/db partials on mid shapes.
    # fp32 live tiles overflow UB (200KB > 192KB); bf16/fp16 (the training path) fit.
    bt_cap = 32 if x.dtype == torch.float32 else 64
    BT = min(bt_cap, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
        NUM_CHKS = len(chunk_indices)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, BT)
        NUM_CHKS = NT * B
    has_parallelism = triton.cdiv(D, 64) * NUM_CHKS > NUM_CORES // 2
    preferred = 64 if has_parallelism else 32
    BD = _select_coregrid_bd(D, preferred)
    if BD is None:
        raise RuntimeError(f'coregrid bwd: no aligned BD for D={D}')
    NUM_BLKS_D = triton.cdiv(D, BD)

    if y_linear is None and activation is not None:
        y_linear, _ = _launch_fwd_coregrid(
            x, weight, bias, None,
            activation=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
    dx = torch.empty_like(x)
    dw = weight.new_empty(B * NT, W, D, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B * NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None

    def _invoke(num_chks: int, nt_stride: int, i_t_offset: int, tail_mode: int) -> None:
        causal_conv1d_bwd_coregrid_kernel[(NUM_CORES,)](
            x=x,
            y=y_linear,
            weight=weight_wd,
            initial_state=None,
            dh0=None,
            dht=None,
            dy=dy,
            dx=dx,
            dw=dw,
            db=db,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            B=B,
            T=T,
            D=D,
            W=W,
            BT=BT,
            BD=BD,
            ACTIVATION=activation,
            NUM_BLKS_D=NUM_BLKS_D,
            NUM_CHKS=num_chks,
            NT_STRIDE=nt_stride,
            I_T_OFFSET=i_t_offset,
            TAIL_MODE=tail_mode,
        )

    # packed NT>1: constexpr-split the last T-chunk so the bulk kernel DCE's the masked DMA path.
    if cu_seqlens is None and NT > 1:
        _invoke(num_chks=B * (NT - 1), nt_stride=NT - 1, i_t_offset=0, tail_mode=0)
        _invoke(num_chks=B, nt_stride=1, i_t_offset=NT - 1, tail_mode=1)
    else:
        _invoke(num_chks=NUM_CHKS, nt_stride=max(NT, 1), i_t_offset=0, tail_mode=2)
    if weight is not None:
        dw = dw.sum(0).to(weight).transpose(0, 1)
    if bias is not None:
        db = db.sum(0).to(bias)
    return dx, dw, db, dr


def _can_use_coregrid(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    dht: torch.Tensor | None = None,
    layout_fallback: bool = False,
) -> bool:
    """Whether the 1D core-grid kernels can run for this layout and channel width."""
    if layout_fallback or initial_state is not None or dht is not None:
        return False
    if not x.is_contiguous() or x.stride(-1) != 1:
        return False
    if residual is not None and (not residual.is_contiguous() or residual.stride(-1) != 1):
        return False
    D = x.shape[-1]
    # need a usable channel tile (legacy path covers odd D like 200).
    bd = _select_coregrid_bd(D, 64)
    return bd is not None and bd >= 16


@input_guard(no_guard_contiguous=['x'])
def causal_conv1d_fwd_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
    layout_fallback: bool = False,
):
    shape = x.shape
    # bwd reads the stashed y_linear from the pre-rearrange tensor.
    x_saved = x
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    # 1D core-grid fast path (large BD + extract_slice).
    if _can_use_coregrid(x, residual, initial_state, layout_fallback=layout_fallback):
        save_linear = activation in ('swish', 'silu')
        y, y_linear = _launch_fwd_coregrid(
            x, weight, bias, residual, activation, cu_seqlens, cu_seqlens_cpu,
            save_linear=save_linear,
        )
        if y_linear is not None:
            x_saved._fla_causal_conv_y_linear = y_linear
        final_state = None
        if output_final_state:
            final_state = causal_conv1d_update_states_npu(
                x=x,
                state_len=weight.shape[1],
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
            )
        return y.view(shape), final_state

    del layout_fallback
    B, T, D = x.shape[0], x.shape[1], weight.shape[0]
    W = weight.shape[1]

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)

    if initial_state is None:
        # ant_AReaL fused block_ptr kernel (fast): bias/silu/residual fused in.
        y = _launch_fwd_core(
            x, weight, bias, residual, initial_state, cu_seqlens, chunk_indices,
            B, T, D, W, BT, activation,
        )
    else:
        # initial_state present: the ant_AReaL kernel's head-edge branch faults the Ascend
        # vector core on this triton-ascend version, so use the scalar kernel (conv only)
        # and apply activation/residual afterwards.
        y = _launch_fwd_core_scalar(
            x, weight, bias, initial_state, cu_seqlens, cu_seqlens_cpu, B, T, D, W, BT,
        )
        y = _postprocess_fwd(y, residual, activation)

    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states_npu(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )
    return y.view(shape), final_state


def causal_conv1d_bwd_npu(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
    layout_fallback: bool = False,
):
    shape = x.shape
    y_linear = getattr(x, '_fla_causal_conv_y_linear', None)
    if y_linear is not None:
        del x._fla_causal_conv_y_linear
    if weight is not None and x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
        if y_linear is not None and y_linear.shape != x.shape:
            y_linear = rearrange(y_linear, 'b t ... -> b t (...)')
    if _can_use_coregrid(x, residual, initial_state, dht=dht, layout_fallback=layout_fallback):
        # core-grid assumes packed BTD; cheap-copy strided dy from cat/split views.
        if dy is not None and not dy.is_contiguous():
            dy = dy.contiguous()
        dx, dw, db, dr = _launch_bwd_coregrid(
            x, dy, weight, bias, residual, activation, cu_seqlens, cu_seqlens_cpu,
            y_linear=y_linear,
        )
        return dx.view(shape), dw, db, dr, None

    del layout_fallback
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None

    # dw/db kernel overflows UB on non-contiguous x; ensure contiguous (QKV views pay a copy).
    if not x.is_contiguous():
        x = x.contiguous()

    dr = dy if residual is not None else None
    dy_conv = dy

    y_pre = None
    if activation in ('swish', 'silu'):
        if initial_state is None:
            chunk_indices_f = None
            if cu_seqlens is not None:
                chunk_indices_f = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
            y_pre = _launch_fwd_core(
                x, weight, bias, None, initial_state, cu_seqlens, chunk_indices_f,
                B, T, D, W, BT, activation=None,
            )
        else:
            y_pre = _launch_fwd_core_scalar(
                x, weight, bias, initial_state, cu_seqlens, cu_seqlens_cpu, B, T, D, W, BT,
            )
        dy_conv = _launch_silu_bwd(y_pre, dy)

    use_seq = _use_seq_bwd(T, x.dtype, initial_state, dht, cu_seqlens)

    if use_seq:
        # tiny-T path (T <= 16): the combined seq kernel is cheap enough; keep it.
        stride_x_n, stride_x_t, stride_x_d = x.stride()
        if not dy_conv.is_contiguous():
            dy_conv = dy_conv.contiguous()
        stride_dy_n, stride_dy_t, stride_dy_d = dy_conv.stride()
        dx = torch.zeros_like(x)
        stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()
        block = 1024
        dw = weight.new_empty(B * T, *weight.shape, dtype=torch.float) if weight is not None else None
        db = bias.new_empty(B * T, *bias.shape, dtype=torch.float) if bias is not None else None
        grid = (triton.cdiv(B * T * D, block),)
        causal_conv1d_bwd_seq_kernel[grid](
            x=x,
            weight=weight,
            dy=dy_conv,
            dx=dx,
            dw=dw,
            db=db,
            stride_x_n=stride_x_n,
            stride_x_t=stride_x_t,
            stride_x_d=stride_x_d,
            stride_dx_n=stride_dx_n,
            stride_dx_t=stride_dx_t,
            stride_dx_d=stride_dx_d,
            stride_dy_n=stride_dy_n,
            stride_dy_t=stride_dy_t,
            stride_dy_d=stride_dy_d,
            B=B,
            TC=T,
            D=D,
            W=W,
            BLOCK=block,
        )
    else:
        # Split backward: dx (big tile, ~fwd speed) + dw/db (small tile). The old combined
        # kernel was pinned to tiny tiles by the MLIR/multi-buffer UB cliff.
        if not dy_conv.is_contiguous():
            dy_conv = dy_conv.contiguous()
        dx = _launch_bwd_dx_core(
            dy_conv, weight, dht, cu_seqlens, cu_seqlens_cpu,
            B, T, D, W, BT,
        )
        if weight is not None or bias is not None:
            dw, db = _launch_bwd_dwdb_core(
                x, dy_conv, initial_state, cu_seqlens, cu_seqlens_cpu,
                B, T, D, W, BT, weight=weight, bias=bias,
            )
        else:
            dw, db = None, None
    if weight is not None:
        dw = dw.sum(0).to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)

    dh0 = None
    if initial_state is not None:
        dh0 = compute_dh0_npu(
            dy=dy,
            y=y_pre,
            weight=weight,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
        )

    return dx.view(shape), dw, db, dr, dh0


def compute_dh0_npu(
    dy: torch.Tensor,
    y: torch.Tensor | None,
    weight: torch.Tensor,
    initial_state: torch.Tensor,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    D, W = weight.shape
    N = initial_state.shape[0]
    T = dy.shape[1]

    BD = 8 if dy.dtype == torch.float16 and activation in ('swish', 'silu') else 16
    dh0 = torch.zeros_like(initial_state)

    stride_dy_n = dy.stride(0)
    stride_dy_t = dy.stride(1)
    stride_dy_d = dy.stride(2) if dy.dim() == 3 else dy.stride(-1)
    stride_y_n = stride_y_t = stride_y_d = 0
    if y is not None:
        stride_y_n = y.stride(0)
        stride_y_t = y.stride(1)
        stride_y_d = y.stride(2) if y.dim() == 3 else y.stride(-1)

    max_n = _npu_max_axis_chunks(triton.cdiv(D, BD))
    kernel_kwargs = dict(
        dy=dy,
        y=y if activation in ('swish', 'silu') else None,
        weight=weight,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        stride_dy_n=stride_dy_n,
        stride_dy_t=stride_dy_t,
        stride_dy_d=stride_dy_d,
        stride_y_n=stride_y_n,
        stride_y_t=stride_y_t,
        stride_y_d=stride_y_d,
        T=T,
        D=D,
        W=W,
        BD=BD,
    )
    for n_off in range(0, N, max_n):
        n_len = min(max_n, N - n_off)
        kernel_kwargs['CHUNK_OFFSET'] = n_off
        compute_dh0_kernel[(triton.cdiv(D, BD), n_len)](**kernel_kwargs)
    return dh0


@input_guard(no_guard_contiguous=['x'])
def causal_conv1d_update_states_npu(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    layout_fallback: bool = False,
) -> torch.Tensor:
    del layout_fallback
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if x.dim() == 2:
            stride_x_n = 0
            stride_x_t, stride_x_d = x.stride()
            T = x.shape[0]
        else:
            stride_x_n = x.stride(0)
            stride_x_t, stride_x_d = x.stride(1), x.stride(2)
            T = x.shape[1]
        D = x.shape[-1]
    else:
        B, T, D = x.shape
        N = B
        stride_x_n, stride_x_t, stride_x_d = x.stride()

    W = state_len
    final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)
    BD = min(triton.next_power_of_2(D), 16)
    BW = triton.next_power_of_2(W)
    grid_dim0 = triton.cdiv(D, BD)
    max_n = _npu_max_axis_chunks(grid_dim0)
    kernel_kwargs = dict(
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        BW=BW,
        BD=BD,
    )
    for n_off in range(0, N, max_n):
        n_len = min(max_n, N - n_off)
        kernel_kwargs['CHUNK_OFFSET'] = n_off
        causal_conv1d_states_fwd_kernel[(grid_dim0, n_len)](**kernel_kwargs)
    return final_state


@input_guard(no_guard_contiguous=['x'])
def causal_conv1d_update_npu(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    if weight is not None and x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')

    D = x.shape[-1]
    N = x.numel() // D
    W = weight.shape[1] if weight is not None else None
    BD = min(triton.next_power_of_2(D), 16)

    if x.dim() == 2:
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(1)
    elif x.dim() == 3 and x.shape[0] == 1:
        stride_x_n = x.stride(1)
        stride_x_d = x.stride(2)
    elif x.dim() == 3:
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(2)
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")

    y = torch.zeros_like(x, memory_format=torch.contiguous_format)

    if y.dim() == 2:
        stride_y_n, stride_y_d = y.stride(0), y.stride(1)
    elif y.dim() == 3 and y.shape[0] == 1:
        stride_y_n, stride_y_d = y.stride(1), y.stride(2)
    elif y.dim() == 3:
        stride_y_n, stride_y_d = y.stride(0), y.stride(2)

    grid_dim0 = triton.cdiv(D, BD)
    max_n = _npu_max_axis_chunks(grid_dim0)
    kernel_kwargs = dict(
        x=x,
        cache=cache,
        y=y,
        weight=weight,
        bias=bias,
        stride_x_n=stride_x_n,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_d=stride_y_d,
        D=D,
        W=W,
        BD=BD,
    )
    for n_off in range(0, N, max_n):
        n_len = min(max_n, N - n_off)
        kernel_kwargs['CHUNK_OFFSET'] = n_off
        causal_conv1d_update_kernel[(grid_dim0, n_len)](**kernel_kwargs)
    y = _postprocess_update(y, residual, activation)
    return y.view(shape), cache
