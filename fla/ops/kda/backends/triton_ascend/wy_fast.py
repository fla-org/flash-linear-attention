# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA WY-representation kernels adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

_NUM_WARPS = 2
# recompute_w_u_fwd: b_A[BT,BT], b_vb[BT,BV], b_kb[BT,BK]
_RECOMPUTE_FWD_MEM_MULT = 6.0
_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 8
_MAX_TILE_FWD = 64


def _get_fwd_tiles(BT: int, K: int, V: int) -> tuple[int, int]:
    BK = compute_row_tile_block_size(
        BT, K, _RECOMPUTE_FWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=min(_MAX_TILE_FWD, triton.next_power_of_2(K)),
    )
    BV = compute_row_tile_block_size(
        BT, V, _RECOMPUTE_FWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=min(_MAX_TILE_FWD, triton.next_power_of_2(V)),
    )
    return BK, BV


def _launch_wy_kernel(kernel, *, NT: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = max_grid_axis_chunks(NT, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](num_warps=_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kda_kernel_npu(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    u += (bos * HV + i_hv) * V
    w += (bos * HV + i_hv) * K
    gk += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    kg += (bos * HV + i_hv) * K
    if STORE_QG:
        q += (bos * H + i_h) * K
        qg += (bos * HV + i_hv) * K

    p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_vb = b_v * b_b[:, None]
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    last_idx = min(i_t * BT + BT, T) - 1
    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(gk, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb = b_kb * exp2(b_gk)

        if STORE_QG:
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_qg = tl.make_block_ptr(qg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))

        if STORE_KG:
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(gk + last_idx * HV * K + o_k, mask=m_k, other=0.).to(tl.float32)
            b_kg = b_k * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg = tl.make_block_ptr(kg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(tl.float32), allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def recompute_w_u_fwd_kda_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    q: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BT = A.shape[-1]
    BK, BV = _get_fwd_tiles(BT, K, V)
    store_qg = q is not None
    is_varlen = cu_seqlens is not None

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.zeros(B, T, HV, K, device=k.device, dtype=k.dtype)
    u = torch.zeros_like(v)
    qg = torch.zeros(B, T, HV, K, device=k.device, dtype=k.dtype) if store_qg else None
    kg = torch.zeros(B, T, HV, K, device=k.device, dtype=k.dtype)

    _launch_wy_kernel(
        recompute_w_u_fwd_kda_kernel_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            q=q,
            k=k,
            qg=qg,
            kg=kg,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            gk=gk,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            STORE_QG=store_qg,
            STORE_KG=True,
            IS_VARLEN=is_varlen,
        ),
    )
    return w, u, qg, kg
