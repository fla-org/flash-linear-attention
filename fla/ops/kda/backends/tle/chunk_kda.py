# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright 2026, The FlagOS Contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
"""TLE forward path for KDA chunk inference."""

from __future__ import annotations

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.runtime._allocation import NullAllocator, _allocator
from triton.tools.tensor_descriptor import TensorDescriptor

from fla.ops.kda.backends.tle import _tle_input_error
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import autotune_cache_kwargs, input_guard
from fla.utils._device import _default_alloc_fn

__all__ = ["chunk_kda_fwd_infer"]


RCP_LN2 = 1.4426950216


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages, maxnreg=maxnreg)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 4, 8]
        for maxnreg in [None, 32, 64, 72]
    ],
    key=["H", "K", "BT", "IS_VARLEN", "HAS_DT_BIAS"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_intra_kernel(
    q,
    k,
    g,
    beta,
    ws,
    Aqk,
    Akk,
    g_last,
    A_log,
    dt_bias,
    lower_bound,
    scale,
    g_scale,
    l2norm_eps,
    cu_seqlens,
    chunk_indices,
    T,
    NT_TOTAL,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
):
    i_chunk_pid = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_h = i_bh % H

    if IS_VARLEN:
        i_chunk_global = i_chunk_pid
        i_n = tl.load(chunk_indices + i_chunk_pid * 2).to(tl.int64)
        i_chunk = tl.load(chunk_indices + i_chunk_pid * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_bh // H
        i_chunk = i_chunk_pid
        bos = i_n.to(tl.int64) * T
        i_chunk_global = i_n * tl.cdiv(T, BT) + i_chunk

    if i_chunk * BT >= T:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    g_last += (i_chunk_global * H + i_h).to(tl.int64) * K
    if IS_VARLEN:
        a_chunk = i_h * NT_TOTAL + i_chunk_global
    else:
        a_chunk = (i_n * H + i_h) * NT_TOTAL + i_chunk
    Aqk += a_chunk.to(tl.int64) * BT * BT
    Akk += a_chunk.to(tl.int64) * BT * BT
    ws += (bos * H + i_h) * 3 * K
    beta += bos * H + i_h

    o_i = tl.arange(0, BT)
    o_i64 = o_i.to(tl.int64)
    o_k = tl.arange(0, K)
    o_k64 = o_k.to(tl.int64)
    token_start = i_chunk * BT
    m_c = token_start + o_i < T

    q_buf = tle.gpu.alloc([BT, K], dtype=q.dtype.element_ty, scope=tle.gpu.smem)
    k_buf = tle.gpu.alloc([BT, K], dtype=k.dtype.element_ty, scope=tle.gpu.smem)
    gc_buf = tle.gpu.alloc([BT, K], dtype=tl.float32, scope=tle.gpu.smem)

    rows = tl.broadcast_to(o_i[:, None], (BT, K))
    cols = tl.broadcast_to(o_k[None, :], (BT, K))
    q_sp = tle.gpu.local_ptr(q_buf, (rows, cols))
    k_sp = tle.gpu.local_ptr(k_buf, (rows, cols))
    gc_sp = tle.gpu.local_ptr(gc_buf, (rows, cols))

    offsets_qkg = (token_start + o_i64)[:, None] * H * K + o_k64[None, :]
    b_q = tl.load(q + offsets_qkg, mask=m_c[:, None], other=0.0)
    b_k = tl.load(k + offsets_qkg, mask=m_c[:, None], other=0.0)
    tl.store(q_sp, b_q)
    tl.store(k_sp, b_k)

    b_qf = b_q.to(tl.float32)
    b_kf = b_k.to(tl.float32)

    b_q_rstd = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + l2norm_eps)
    b_k_rstd = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + l2norm_eps)

    b_g = tl.load(g + offsets_qkg, mask=m_c[:, None], other=0.0).to(tl.float32)
    b_A = exp2(tl.load(A_log + i_h).to(tl.float32) * g_scale)
    if HAS_DT_BIAS:
        b_bias = tl.load(dt_bias + i_h * K + o_k64).to(tl.float32)
        b_g += b_bias[None, :]
    b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)
    tl.store(gc_sp, b_g)
    one_row = tl.broadcast_to(tl.arange(0, 1)[:, None], (1, K))
    col_row = tl.broadcast_to(tl.arange(0, K)[None, :], (1, K))
    b_acc = tl.zeros([1, K], dtype=tl.float32)
    for r in tl.static_range(BT):
        rp = tle.gpu.local_ptr(gc_buf, (tl.broadcast_to(one_row + r, (1, K)), col_row))
        b_acc = b_acc + tl.load(rp)
        tl.store(rp, b_acc)

    b_gq = tl.where(m_c[:, None], exp2(tl.load(gc_sp)), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-tl.load(gc_sp)), 0.0)

    b_kgt = tl.trans(b_kf * b_gk).to(b_k.dtype)
    b_Aqk = tl.dot(
        (b_qf * b_gq).to(b_q.dtype),
        b_kgt,
        out_dtype=tl.float32,
    )
    b_Akk = tl.dot(
        (b_kf * b_gq).to(b_k.dtype),
        b_kgt,
        out_dtype=tl.float32,
    )

    b_Aqk = b_Aqk * b_q_rstd[:, None] * b_k_rstd[None, :]
    b_Akk = b_Akk * b_k_rstd[:, None] * b_k_rstd[None, :]

    b_beta = tl.sigmoid(tl.load(beta + (token_start + o_i64) * H, mask=m_c, other=0.0).to(tl.float32))

    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

    offsets_A = o_i64[:, None] * BT + o_i64[None, :]
    tl.store(Aqk + offsets_A, b_Aqk.to(Aqk.dtype.element_ty))

    b_L = b_Akk.to(tl.float16)
    b_Ai = m_I.to(tl.float16) - b_L
    b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L2, out_dtype=tl.float16)
    b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L4, out_dtype=tl.float16)
    b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L8, out_dtype=tl.float16)

    tl.store(Akk + offsets_A, b_Ai.to(Akk.dtype.element_ty))

    b_k3 = tl.load(k_sp).to(tl.float32) * b_k_rstd[:, None]
    b_gk3 = tl.load(gc_sp)
    b_kb = b_k3 * b_beta[:, None] * exp2(b_gk3)
    offsets_ws = (token_start + o_i64)[:, None] * H * 3 * K + o_k64[None, :]
    tl.store(ws + offsets_ws, b_kb.to(ws.dtype.element_ty), mask=m_c[:, None])

    b_qg_val = tl.load(q_sp).to(tl.float32) * b_q_rstd[:, None] * exp2(b_gk3)
    tl.store(ws + offsets_ws + K, b_qg_val.to(ws.dtype.element_ty), mask=m_c[:, None])

    last_local = (tl.minimum(BT, T - token_start) - 1).to(tl.int32)
    gn_rows = tl.broadcast_to(last_local + tl.zeros([1, K], dtype=tl.int32), (1, K))
    gn_cols = tl.broadcast_to(tl.arange(0, K)[None, :], (1, K))
    b_gn = tl.load(tle.gpu.local_ptr(gc_buf, (gn_rows, gn_cols)))
    tl.store(g_last + o_k64, b_gn.reshape([K]).to(g_last.dtype.element_ty))
    b_kg_val = b_k3 * tl.where(m_c[:, None], exp2(b_gn - b_gk3), 0)
    tl.store(ws + offsets_ws + 2 * K, b_kg_val.to(ws.dtype.element_ty), mask=m_c[:, None])


def _kda_fwd_intra(
    q,
    k,
    g,
    beta,
    scale,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=16,
    lower_bound=None,
    A_log=None,
    dt_bias=None,
):
    B, T_len, H, K = q.shape
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T_len, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, B * H)

    g_last = torch.empty(B * NT, H, K, device=q.device, dtype=torch.float32)
    ws = torch.empty(B, NT * BT, H, 3 * K, device=q.device, dtype=q.dtype)
    Aqk = torch.empty(B, H, NT, BT, BT, device=q.device, dtype=q.dtype)
    Akk = torch.empty(B, H, NT, BT, BT, device=q.device, dtype=q.dtype)

    _kda_fwd_intra_kernel[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        ws=ws,
        Aqk=Aqk,
        Akk=Akk,
        g_last=g_last,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        scale=scale,
        g_scale=RCP_LN2,
        l2norm_eps=1e-6,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T_len,
        NT_TOTAL=NT,
        H=H,
        K=K,
        BT=BT,
    )
    return ws, Aqk, Akk, g_last


@triton.jit
def _kda_state_output_load_producer(
    writer,
    ws_desc,
    v_ptr,
    beta_ptr,
    gk_desc,
    Aqk_desc,
    Akk_desc,
    K: tl.constexpr,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT,
    i_v,
    USE_HOST_DESCRIPTORS: tl.constexpr,
    state_row_start,
    chunk_start,
    i_h,
):
    for i_chunk in tl.range(NT):
        slot = writer.acquire(i_chunk)

        ws_row = i_chunk * BT
        ws_col = 0
        A_row = i_chunk * BT
        gk_row = i_chunk
        gk_col = 0
        if USE_HOST_DESCRIPTORS:
            ws_row += state_row_start
            ws_col = i_h * 3 * K
            A_row = state_row_start * H + i_h * NT * BT + i_chunk * BT
            gk_row += chunk_start
            gk_col = i_h * K

        ws_row = ws_row.to(tl.int32)
        ws_col = ws_col.to(tl.int32)
        A_row = A_row.to(tl.int32)
        gk_row = gk_row.to(tl.int32)
        gk_col = gk_col.to(tl.int32)

        tle.gpu.copy(ws_desc, slot.w, [BT, K], [ws_row, ws_col])
        tle.gpu.copy(ws_desc, slot.qg, [BT, K], [ws_row, ws_col + K])
        tle.gpu.copy(ws_desc, slot.kg, [BT, K], [ws_row, ws_col + 2 * K])
        tle.gpu.copy(Aqk_desc, slot.Aqk, [BT, BT], [A_row, 0])
        tle.gpu.copy(Akk_desc, slot.Akk, [BT, BT], [A_row, 0])
        tle.gpu.copy(gk_desc, slot.gk, [1, K], [gk_row, gk_col])

        o_t = tl.arange(0, BT)
        o_t64 = o_t.to(tl.int64)
        o_v = tl.arange(0, BV)
        o_v64 = o_v.to(tl.int64)
        token = i_chunk.to(tl.int64) * BT + o_t64
        value = i_v.to(tl.int64) * BV + o_v64
        mask_t = token < T
        b_v_raw = tl.load(
            v_ptr + token[:, None] * H * V + value[None, :],
            mask=mask_t[:, None] & (value[None, :] < V),
            other=0.0,
        )
        b_beta = tl.sigmoid(tl.load(beta_ptr + token * H, mask=mask_t, other=0.0).to(tl.float32))
        b_v = (b_v_raw.to(tl.float32) * b_beta[:, None]).to(b_v_raw.dtype)
        tl.store(tle.gpu.local_ptr(slot.v), b_v)

        writer.commit(i_chunk)


@triton.jit
def _kda_state_output_mma_consumer(
    load_reader,
    store_writer,
    h0,
    ht,
    scale,
    i_v,
    i_nh,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NT,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    state_dtype: tl.constexpr = tl.bfloat16
    if USE_INITIAL_STATE:
        o_v = i_v.to(tl.int64) * BV + tl.arange(0, BV).to(tl.int64)
        o_k = tl.arange(0, K).to(tl.int64)
        offsets_h = i_nh.to(tl.int64) * K * V + o_v[:, None] * K + o_k[None, :]
        b_h = tl.trans(tl.load(h0 + offsets_h, mask=o_v[:, None] < V, other=0.0)).to(tl.float32)
    else:
        b_h = tl.zeros([K, BV], dtype=tl.float32)

    for i_chunk in tl.range(NT):
        wait = load_reader.wait(i_chunk)
        slot = wait.slot

        b_w = tl.load(tle.gpu.local_ptr(slot.w))
        b_v_raw = tl.load(tle.gpu.local_ptr(slot.v))
        b_qg = tl.load(tle.gpu.local_ptr(slot.qg))
        b_kg = tl.load(tle.gpu.local_ptr(slot.kg))
        b_Aqk = tl.load(tle.gpu.local_ptr(slot.Aqk))
        b_Akk = tl.load(tle.gpu.local_ptr(slot.Akk))
        b_gk = tl.load(tle.gpu.local_ptr(slot.gk)).reshape([K])

        b_h_bf = b_h.to(state_dtype)

        b_kh = tl.dot(b_w, b_h_bf).to(tl.float32)
        b_diff = b_v_raw.to(tl.float32) - b_kh
        b_v = tl.dot(b_Akk, b_diff.to(state_dtype)).to(tl.float32)

        b_qh = tl.dot(b_qg, b_h_bf).to(tl.float32)
        b_o = scale * b_qh
        b_v_cast = b_v.to(state_dtype)
        b_o += tl.dot(b_Aqk, b_v_cast).to(tl.float32)

        out_slot = store_writer.acquire(i_chunk)
        tl.store(tle.gpu.local_ptr(out_slot.output), b_o)
        store_writer.commit(i_chunk)

        load_reader.release(i_chunk)

        b_h = b_h * exp2(b_gk)[:, None] + tl.dot(tl.trans(b_kg), b_v_cast).to(tl.float32)

    if STORE_FINAL_STATE:
        o_v = i_v.to(tl.int64) * BV + tl.arange(0, BV).to(tl.int64)
        o_k = tl.arange(0, K).to(tl.int64)
        offsets_ht = i_nh.to(tl.int64) * K * V + o_v[:, None] * K + o_k[None, :]
        tl.store(ht + offsets_ht, tl.trans(b_h).to(ht.dtype.element_ty), mask=o_v[:, None] < V)


@triton.jit
def _kda_state_output_store_consumer(
    store_reader,
    store_target,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT,
    i_v,
    USE_HOST_DESCRIPTORS: tl.constexpr,
    output_row_start,
    output_col_start,
):
    for i_chunk in tl.range(NT):
        store_wait = store_reader.wait(i_chunk)
        slot = store_wait.slot
        output_row = i_chunk * BT
        output_col = i_v * BV
        if USE_HOST_DESCRIPTORS:
            output_row += output_row_start
            output_col += output_col_start
        output_row = output_row.to(tl.int32)
        output_col = output_col.to(tl.int32)
        tle.gpu.copy(slot.output, store_target, [BT, BV], [output_row, output_col])
        store_reader.release(i_chunk)


PIPE_STAGES = tl.constexpr(4)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
        "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_state_output_kernel(
    v,
    beta,
    gk,
    Aqk,
    Akk,
    o,
    ws,
    h0,
    ht,
    ws_host_desc,
    gk_host_desc,
    Aqk_host_desc,
    Akk_host_desc,
    output_host_desc,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    NT_TOTAL,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_HOST_DESCRIPTORS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_nh = tl.program_id(1).to(tl.int64)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
        chunk_start = tl.load(chunk_offsets + i_n).to(tl.int32)
        state_row_start = bos
    else:
        bos = i_n.to(tl.int64) * T
        NT = tl.cdiv(T, BT)
        chunk_start = i_n * NT
        state_row_start = i_n.to(tl.int64) * NT * BT

    v += (bos * H + i_h) * V
    beta += bos * H + i_h
    gk += (chunk_start * H + i_h).to(tl.int64) * K
    o += (bos * H + i_h) * V
    ws_base = ws + (bos * H + i_h) * 3 * K

    if USE_HOST_DESCRIPTORS:
        ws_desc = ws_host_desc
        gk_desc = gk_host_desc
        Aqk_desc = Aqk_host_desc
        Akk_desc = Akk_host_desc
    else:
        if IS_VARLEN:
            a_chunk = (i_h * NT_TOTAL + chunk_start) * BT * BT
        else:
            a_chunk = (i_n * H + i_h) * NT_TOTAL * BT * BT
        Aqk += a_chunk.to(tl.int64)
        Akk += a_chunk.to(tl.int64)
        ws_desc = tl.make_tensor_descriptor(ws_base, shape=[T, 3 * K], strides=[H * 3 * K, 1], block_shape=[BT, K])
        gk_desc = tl.make_tensor_descriptor(gk, shape=[NT, K], strides=[H * K, 1], block_shape=[1, K])
        Aqk_desc = tl.make_tensor_descriptor(Aqk, shape=[NT * BT, BT], strides=[BT, 1], block_shape=[BT, BT])
        Akk_desc = tl.make_tensor_descriptor(Akk, shape=[NT * BT, BT], strides=[BT, 1], block_shape=[BT, BT])

    w_smem = tle.gpu.alloc([PIPE_STAGES, BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem)
    v_smem = tle.gpu.alloc([PIPE_STAGES, BT, BV], dtype=tl.bfloat16, scope=tle.gpu.smem)
    qg_smem = tle.gpu.alloc([PIPE_STAGES, BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem)
    kg_smem = tle.gpu.alloc([PIPE_STAGES, BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem)
    Aqk_smem = tle.gpu.alloc([PIPE_STAGES, BT, BT], dtype=tl.bfloat16, scope=tle.gpu.smem)
    Akk_smem = tle.gpu.alloc([PIPE_STAGES, BT, BT], dtype=tl.bfloat16, scope=tle.gpu.smem)
    gk_smem = tle.gpu.alloc([PIPE_STAGES, 1, K], dtype=tl.float32, scope=tle.gpu.smem)
    out_smem = tle.gpu.alloc([PIPE_STAGES, BT, BV], dtype=tl.bfloat16, scope=tle.gpu.smem)
    if USE_HOST_DESCRIPTORS:
        output_store_target = output_host_desc
    else:
        output_store_target = tl.make_tensor_descriptor(
            o,
            shape=[T, V],
            strides=[H * V, 1],
            block_shape=[BT, BV],
        )

    load_pipe = tle.pipe(
        capacity=PIPE_STAGES,
        scope="cta",
        name="kda_load",
        w=w_smem,
        v=v_smem,
        qg=qg_smem,
        kg=kg_smem,
        Aqk=Aqk_smem,
        Akk=Akk_smem,
        gk=gk_smem,
    )
    store_pipe = tle.pipe(
        capacity=PIPE_STAGES,
        scope="cta",
        name="kda_store",
        output=out_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _kda_state_output_load_producer,
                (
                    load_pipe.writer(),
                    ws_desc,
                    v,
                    beta,
                    gk_desc,
                    Aqk_desc,
                    Akk_desc,
                    K,
                    T,
                    H,
                    V,
                    BT,
                    BV,
                    NT,
                    i_v,
                    USE_HOST_DESCRIPTORS,
                    state_row_start,
                    chunk_start,
                    i_h,
                ),
            ),
            (
                _kda_state_output_mma_consumer,
                (
                    load_pipe.reader(),
                    store_pipe.writer(),
                    h0,
                    ht,
                    scale,
                    i_v,
                    i_nh,
                    H,
                    K,
                    V,
                    BV,
                    NT,
                    USE_INITIAL_STATE,
                    STORE_FINAL_STATE,
                ),
            ),
            (
                _kda_state_output_store_consumer,
                (
                    store_pipe.reader(),
                    output_store_target,
                    BT,
                    BV,
                    NT,
                    i_v,
                    USE_HOST_DESCRIPTORS,
                    bos.to(tl.int64),
                    i_h.to(tl.int64) * V,
                ),
            ),
        ],
        [4, 1],
        [240, 32],
    )


def _kda_fwd_state_output(
    v: torch.Tensor,
    beta: torch.Tensor,
    Akk: torch.Tensor,
    gk: torch.Tensor,
    Aqk: torch.Tensor,
    ws: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, _, H, packed_K = ws.shape
    K = packed_K // 3
    T_actual = v.shape[1]
    V = v.shape[-1]
    BT = chunk_size

    if cu_seqlens is None:
        N = B
        chunk_offsets = None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    final_state = ws.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None

    o = torch.empty(B, T_actual, H, V, device=ws.device, dtype=v.dtype)

    h0_arg = initial_state if initial_state is not None else ws.new_empty(1, dtype=torch.float32)
    ht_arg = final_state if final_state is not None else ws.new_empty(1, dtype=torch.float32)

    use_host_descriptors = cu_seqlens is None and T_actual % BT == 0
    if use_host_descriptors:
        NT_total = ws.shape[1] // BT
        descriptor_rows = B * H * NT_total * BT
        ws_desc_arg = TensorDescriptor(
            ws,
            shape=[descriptor_rows, H * 3 * K],
            strides=[H * 3 * K, 1],
            block_shape=[BT, K],
        )
        gk_desc_arg = TensorDescriptor(
            gk,
            shape=[gk.shape[0], H * K],
            strides=[H * K, 1],
            block_shape=[1, K],
        )
        Aqk_desc_arg = TensorDescriptor(
            Aqk,
            shape=[descriptor_rows, BT],
            strides=[BT, 1],
            block_shape=[BT, BT],
        )
        Akk_desc_arg = TensorDescriptor(
            Akk,
            shape=[descriptor_rows, BT],
            strides=[BT, 1],
            block_shape=[BT, BT],
        )
        output_desc_arg = TensorDescriptor(
            o,
            shape=[B * T_actual, H * V],
            strides=[H * V, 1],
            block_shape=[BT, 128],
        )
    else:
        ws_desc_arg = ws
        gk_desc_arg = gk
        Aqk_desc_arg = Aqk
        Akk_desc_arg = Akk
        output_desc_arg = o

    _kda_fwd_state_output_kernel[(1, N * H)](
        v=v,
        beta=beta,
        gk=gk,
        Aqk=Aqk,
        Akk=Akk,
        o=o,
        ws=ws,
        h0=h0_arg,
        ht=ht_arg,
        ws_host_desc=ws_desc_arg,
        gk_host_desc=gk_desc_arg,
        Aqk_host_desc=Aqk_desc_arg,
        Akk_host_desc=Akk_desc_arg,
        output_host_desc=output_desc_arg,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T_actual,
        NT_TOTAL=ws.shape[1] // BT,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BV=128,
        USE_HOST_DESCRIPTORS=use_host_descriptors,
        num_warps=4,
    )

    return o, final_state


@input_guard
def chunk_kda_fwd_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 16,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if torch.is_grad_enabled():
        raise RuntimeError("TLE KDA only supports inference mode")
    reason = _tle_input_error(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    if reason is not None:
        raise ValueError(reason)

    if isinstance(_allocator.get(), NullAllocator):
        triton.set_allocator(_default_alloc_fn)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    ws, Aqk, Akk, g_last = _kda_fwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    return _kda_fwd_state_output(
        v=v,
        beta=beta,
        Akk=Akk,
        gk=g_last,
        Aqk=Aqk,
        ws=ws,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
