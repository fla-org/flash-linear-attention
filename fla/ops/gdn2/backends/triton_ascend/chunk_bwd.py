# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN-2 chunk backward kernels for triton-ascend."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.common.backends.triton_ascend.chunk_delta_h import need_dma_mask
from fla.ops.kda.backends.triton_ascend.chunk_bwd import chunk_kda_bwd_kernel_wy_k_part_npu
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size, get_npu_properties

_BC = 16
_BWD_MEM_MULT = 10.0
_SAFETY_MARGIN = 0.80
_FALLBACK_TILE = 16
_MAX_TILE = 128


def _get_tile(size: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        size,
        _BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=16,
        max_block=min(_MAX_TILE, triton.next_power_of_2(size)),
    )


def _t_contig_arg(x: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, bool]:
    if num_heads == 1:
        return x, False
    return x.transpose(1, 2).contiguous(), True


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_gdn2_bwd_kernel_wy_v_part_npu(
    v,
    w_gate,
    A,
    dv,
    dv2,
    dw,
    dA_acc,
    cu_seqlens,
    chunk_indices,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    T_CONTIG: tl.constexpr,
):
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
                chunk_indices + i_t * 2 + 1,
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            bos = tl.cast(i_b, tl.int64) * T

        if T_CONTIG:
            if IS_VARLEN:
                head_off = tl.cast(i_h, tl.int64) * T_seq
                v_ptr = v + (head_off + bos) * V
                w_ptr = w_gate + (head_off + bos) * V
                dv_ptr = dv + (head_off + bos) * V
                A_ptr = A + (head_off + bos) * BT
            else:
                head_off = (tl.cast(i_b, tl.int64) * H + i_h) * T_seq
                v_ptr = v + head_off * V
                w_ptr = w_gate + head_off * V
                dv_ptr = dv + head_off * V
                A_ptr = A + head_off * BT
            value_stride_t = V
            a_stride_t = BT
        else:
            v_ptr = v + (bos * H + i_h) * V
            w_ptr = w_gate + (bos * H + i_h) * V
            dv_ptr = dv + (bos * H + i_h) * V
            A_ptr = A + (bos * H + i_h) * BT
            value_stride_t = H * V
            a_stride_t = H * BT

        dv2_ptr = dv2 + (bos * H + i_h) * V
        dw_ptr = dw + (bos * H + i_h) * V
        dA_ptr = dA_acc + (bos * H + i_h) * BT
        o_i = tl.arange(0, BT)
        m_i = (o_i >= 0) & (o_i < BT)
        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        m_t = (o_t >= 0) & (o_t < T)
        m_p_A = m_i[:, None] & m_t[None, :]
        p_A = A_ptr + o_i[:, None] + o_t[None, :] * a_stride_t
        b_A = tl.load(p_A, mask=m_p_A, other=0)
        b_dA = tl.zeros([BT, BT], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = (o_v >= 0) & (o_v < V)
            m_p_v = m_t[:, None] & m_v[None, :]
            p_v = v_ptr + o_t[:, None] * value_stride_t + o_v[None, :]
            p_w = w_ptr + o_t[:, None] * value_stride_t + o_v[None, :]
            p_dv = dv_ptr + o_t[:, None] * value_stride_t + o_v[None, :]
            b_v = tl.load(p_v, mask=m_p_v, other=0)
            b_w = tl.load(p_w, mask=m_p_v, other=0)
            b_dv = tl.load(p_dv, mask=m_p_v, other=0)
            # preserve dv for the rhs dot before the first Ascend tl.dot clobbers its lhs
            b_dv_for_dvb = b_dv + 0.0
            b_dA += tl.dot(
                b_dv.to(tl.float32),
                tl.trans(b_v.to(tl.float32) * b_w.to(tl.float32)),
                allow_tf32=False,
            )
            # give each V slab a disposable A lhs because Ascend tl.dot clobbers it
            # A may be fp32 while dv stays in the activation dtype.
            b_A_for_dvb = (b_A + 0.0).to(tl.float32)
            b_dvb = tl.dot(b_A_for_dvb, b_dv_for_dvb.to(tl.float32), allow_tf32=False)

            p_dv2 = dv2_ptr + o_t[:, None] * (H * V) + o_v[None, :]
            p_dw = dw_ptr + o_t[:, None] * (H * V) + o_v[None, :]
            tl.store(p_dv2, (b_dvb * b_w).to(p_dv2.dtype.element_ty), mask=m_p_v)
            tl.store(p_dw, (b_dvb * b_v).to(p_dw.dtype.element_ty), mask=m_p_v)

        m_p_dA = m_t[:, None] & m_i[None, :]
        p_dA = dA_ptr + o_t[:, None] * (H * BT) + o_i[None, :]
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_p_dA)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_gdn2_bwd_kernel_wy_gate_part_npu(
    k,
    g,
    b,
    A,
    h,
    dv,
    dA_acc,
    db,
    dg,
    dk,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    K_T_CONTIG: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    K_OFFSET: tl.constexpr,
):
    i_k = K_OFFSET
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
                chunk_indices + i_t * 2 + 1,
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + tl.cast(i_t, tl.int64)
        else:
            i_tg = tl.cast(i_b, tl.int64) * tl.cdiv(T, BT) + i_t
            bos = tl.cast(i_b, tl.int64) * T

        if K_T_CONTIG:
            if IS_VARLEN:
                k_ptr = k + (tl.cast(i_h, tl.int64) * T_seq + bos) * K
            else:
                k_ptr = k + (tl.cast(i_b, tl.int64) * H + i_h) * T_seq * K
            k_stride_t = K
        else:
            k_ptr = k + (bos * H + i_h) * K
            k_stride_t = H * K

        if G_T_CONTIG:
            if IS_VARLEN:
                head_off = tl.cast(i_h, tl.int64) * T_seq + bos
            else:
                head_off = (tl.cast(i_b, tl.int64) * H + i_h) * T_seq
            g_ptr = g + head_off * K
            b_ptr = b + head_off * K
            A_ptr = A + head_off * BT
            dv_ptr = dv + head_off * V
            g_stride_t = K
            a_stride_t = BT
            dv_stride_t = V
        else:
            g_ptr = g + (bos * H + i_h) * K
            b_ptr = b + (bos * H + i_h) * K
            A_ptr = A + (bos * H + i_h) * BT
            dv_ptr = dv + (bos * H + i_h) * V
            g_stride_t = H * K
            a_stride_t = H * BT
            dv_stride_t = H * V

        h_ptr = h + (i_tg * H + i_h) * K * V
        dA_ptr = dA_acc + (bos * H + i_h) * BT
        db_ptr = db + (bos * H + i_h) * K
        dg_ptr = dg + (bos * H + i_h) * K
        dk_ptr = dk + (bos * H + i_h) * K

        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t = (o_t >= 0) & (o_t < T)
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = (o_v >= 0) & (o_v < V)
            m_p_dv = m_t[:, None] & m_v[None, :]
            p_dv = dv_ptr + o_t[:, None] * dv_stride_t + o_v[None, :]
            if STATE_V_FIRST:
                o_k = i_k * BK + tl.arange(0, BK)
                m_k = (o_k >= 0) & (o_k < K)
                m_p_h = m_v[:, None] & m_k[None, :]
                p_h = h_ptr + o_v[:, None] * K + o_k[None, :]
                b_h = tl.load(p_h, mask=m_p_h, other=0)
            else:
                # [K, V] keeps V contiguous; a strided [V, K] load overflows UB.
                o_k = i_k * BK + tl.arange(0, BK)
                m_k = (o_k >= 0) & (o_k < K)
                m_p_h = m_k[:, None] & m_v[None, :]
                p_h = h_ptr + o_k[:, None] * V + o_v[None, :]
                b_h = tl.trans(tl.load(p_h, mask=m_p_h, other=0))
            b_dv = tl.load(p_dv, mask=m_p_dv, other=0)
            b_dw += tl.dot(b_dv, b_h.to(b_dv.dtype), allow_tf32=False)

        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        m_t = (o_t >= 0) & (o_t < T)
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = (o_k >= 0) & (o_k < K)
        m_p_k = m_t[:, None] & m_k[None, :]
        p_k = k_ptr + o_t[:, None] * k_stride_t + o_k[None, :]
        p_g = g_ptr + o_t[:, None] * g_stride_t + o_k[None, :]
        p_b = b_ptr + o_t[:, None] * g_stride_t + o_k[None, :]
        o_i = tl.arange(0, BT)
        m_i = (o_i >= 0) & (o_i < BT)
        m_p_A = m_i[:, None] & m_t[None, :]
        p_A = A_ptr + o_i[:, None] + o_t[None, :] * a_stride_t
        b_k = tl.load(p_k, mask=m_p_k, other=0)
        b_g = tl.load(p_g, mask=m_p_k, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=m_p_k, other=0)
        b_A = tl.load(p_A, mask=m_p_A, other=0)
        b_gk_exp = exp2(b_g)
        b_kg = b_k * b_gk_exp
        b_dw = -b_dw.to(b_A.dtype)
        b_dkgb = tl.dot(b_A, b_dw, allow_tf32=False)

        m_p_dA = m_t[:, None] & m_i[None, :]
        p_dA = dA_ptr + o_t[:, None] * (H * BT) + o_i[None, :]
        b_dA = tl.load(p_dA, mask=m_p_dA, other=0).to(tl.float32)
        b_dA += tl.dot(b_dw, tl.trans((b_kg * b_b).to(b_A.dtype)), allow_tf32=False)
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_p_dA)

        p_db = db_ptr + o_t[:, None] * (H * K) + o_k[None, :]
        tl.store(p_db, (b_dkgb * b_kg).to(p_db.dtype.element_ty), mask=m_p_k)

        p_dk = dk_ptr + o_t[:, None] * (H * K) + o_k[None, :]
        b_dk = tl.load(p_dk, mask=m_p_k, other=0).to(tl.float32)
        b_dk += b_dkgb * b_gk_exp * b_b
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_p_k)

        p_dg = dg_ptr + o_t[:, None] * (H * K) + o_k[None, :]
        b_dg = tl.load(p_dg, mask=m_p_k, other=0).to(tl.float32)
        b_dg += b_kg * b_dkgb * b_b
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_p_k)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH', 'NT_OFFSET'])
def chunk_gdn2_bwd_kernel_wy_dA_finalize_npu(
    A,
    dA_acc,
    dA,
    cu_seqlens,
    chunk_indices,
    T,
    BH,
    task_num,
    num_core,
    NT_OFFSET,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    A_T_CONTIG: tl.constexpr,
    TAIL_MODE: tl.constexpr,
):
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = NT_OFFSET + task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
                chunk_indices + i_t * 2 + 1,
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            bos = tl.cast(i_b, tl.int64) * T

        if A_T_CONTIG:
            if IS_VARLEN:
                A_ptr = A + (tl.cast(i_h, tl.int64) * T_seq + bos) * BT
            else:
                A_ptr = A + (tl.cast(i_b, tl.int64) * H + i_h) * T_seq * BT
            a_stride_t = BT
        else:
            A_ptr = A + (bos * H + i_h) * BT
            a_stride_t = H * BT

        dA_acc_ptr = dA_acc + (bos * H + i_h) * BT
        dA_ptr = dA + (bos * H + i_h) * BT
        o_i = tl.arange(0, BT)
        m_i = (o_i >= 0) & (o_i < BT)
        o_t_2 = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        m_t_2 = (o_t_2 >= 0) & (o_t_2 < T)
        m_p_A = m_i[:, None] & m_t_2[None, :]
        p_A = A_ptr + o_i[:, None] + o_t_2[None, :] * a_stride_t
        m_p_dA_acc = m_t_2[:, None] & m_i[None, :]
        p_dA_acc = dA_acc_ptr + o_t_2[:, None] * (H * BT) + o_i[None, :]
        p_dA = dA_ptr + o_t_2[:, None] * (H * BT) + o_i[None, :]

        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        if TAIL_MODE == 0:
            b_A = tl.load(p_A)
            b_dA = tl.load(p_dA_acc).to(tl.float32)
            m_A = o_t[:, None] > o_t[None, :]
        else:
            b_A = tl.load(p_A, mask=m_p_A, other=0)
            b_dA = tl.load(p_dA_acc, mask=m_p_dA_acc, other=0).to(tl.float32)
            m_t = o_t < T
            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])

        b_dA = tl.where(m_A, b_dA, 0)
        b_mid = tl.dot(b_dA.to(b_A.dtype), b_A, allow_tf32=False)
        b_fin = tl.dot(b_A, b_mid.to(b_A.dtype), allow_tf32=False)
        b_fin = tl.where(m_A, -b_fin, 0)
        if TAIL_MODE == 0:
            tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty))
        else:
            tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty), mask=m_p_dA_acc)


@input_guard
def chunk_gdn2_bwd_wy_dqkg_fused_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    state_v_first: bool = False,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    if BT % _BC != 0:
        raise ValueError(f'GDN-2 Ascend bwd requires chunk_size % {_BC} == 0, got {BT}')
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = g.new_empty(B, T, H, K, dtype=torch.float)
    dk = g.new_empty(B, T, H, K, dtype=torch.float)
    dv2 = torch.empty(
        v.shape, device=v.device,
        dtype=torch.float32 if dv.dtype == torch.float32 else v.dtype,
    )
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(b, dtype=torch.float)
    dw = torch.empty_like(w_gate, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)
    dA_acc = torch.zeros(B, T, H, BT, dtype=torch.float, device=A.device)

    BK = _get_tile(K)
    BV = _get_tile(V)
    NK = triton.cdiv(K, BK)
    is_varlen = cu_seqlens is not None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if is_varlen else g.new_zeros(1, dtype=torch.int64)
    bh_total = B * H
    task_num = NT * bh_total
    num_core = get_npu_properties()['num_vectorcore']

    v_arg, t_contig = _t_contig_arg(v, H)
    w_arg = _t_contig_arg(w_gate, H)[0]
    A_arg = _t_contig_arg(A, H)[0]
    dv_arg = _t_contig_arg(dv, H)[0]
    chunk_gdn2_bwd_kernel_wy_v_part_npu[(num_core,)](
        v=v_arg,
        w_gate=w_arg,
        A=A_arg,
        dv=dv_arg,
        dv2=dv2,
        dw=dw,
        dA_acc=dA_acc,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
        IS_VARLEN=is_varlen,
        T_CONTIG=t_contig,
    )

    bc_k = 32 if BT >= 32 else _BC
    mask_dma = need_dma_mask(T, BT, K, BK, V, BV, is_varlen) or BT % bc_k != 0
    k_part_kwargs = dict(
        q=q,
        k=k,
        v_new=v_new,
        g=g,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=H,
        K=K,
        V=V,
        BT=BT,
        BC=bc_k,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
        MASK_DMA=mask_dma,
        FP32_H=h.dtype == torch.float32,
    )
    for k_off in range(NK):
        k_part_kwargs['K_OFFSET'] = k_off
        chunk_kda_bwd_kernel_wy_k_part_npu[(num_core,)](**k_part_kwargs)

    k_arg, k_t_contig = _t_contig_arg(k, H)
    g_arg, g_t_contig = _t_contig_arg(g, H)
    b_arg = _t_contig_arg(b, H)[0]
    gate_kwargs = dict(
        k=k_arg,
        g=g_arg,
        b=b_arg,
        A=A_arg,
        h=h,
        dv=dv_arg,
        dA_acc=dA_acc,
        db=db,
        dg=dg,
        dk=dk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
        K_T_CONTIG=k_t_contig,
        G_T_CONTIG=g_t_contig,
    )
    for k_off in range(NK):
        gate_kwargs['K_OFFSET'] = k_off
        chunk_gdn2_bwd_kernel_wy_gate_part_npu[(num_core,)](**gate_kwargs)

    finalize_kwargs = dict(
        A=A_arg,
        dA_acc=dA_acc,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BH=bh_total,
        num_core=num_core,
        H=H,
        BT=BT,
        IS_VARLEN=is_varlen,
        A_T_CONTIG=t_contig,
    )
    if is_varlen:
        finalize_kwargs['task_num'] = NT * bh_total
        finalize_kwargs['NT_OFFSET'] = 0
        finalize_kwargs['TAIL_MODE'] = 1
        chunk_gdn2_bwd_kernel_wy_dA_finalize_npu[(num_core,)](**finalize_kwargs)
    else:
        n_bulk = NT if T % BT == 0 else max(NT - 1, 0)
        if n_bulk > 0:
            finalize_kwargs['task_num'] = n_bulk * bh_total
            finalize_kwargs['NT_OFFSET'] = 0
            finalize_kwargs['TAIL_MODE'] = 0
            chunk_gdn2_bwd_kernel_wy_dA_finalize_npu[(num_core,)](**finalize_kwargs)
        if T % BT != 0 and NT > 0:
            finalize_kwargs['task_num'] = bh_total
            finalize_kwargs['NT_OFFSET'] = n_bulk
            finalize_kwargs['TAIL_MODE'] = 1
            chunk_gdn2_bwd_kernel_wy_dA_finalize_npu[(num_core,)](**finalize_kwargs)
    return dq, dk, dv2, db, dw, dg, dA
