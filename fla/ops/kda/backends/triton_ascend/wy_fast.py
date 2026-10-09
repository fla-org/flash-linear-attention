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
import triton.runtime.driver as driver

from fla.ops.common.backends.triton_ascend.chunk_delta_h import mload, mstore, need_dma_mask
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size

# recompute_w_u_fwd: peak UB is max(u-slab, w-slab), not sum — tile BK/BV independently.
# u-slab: b_A[BT,BT] + b_v/b_vb/b_u[BT,BV]  (~3.5× BT×BV + BT×BT)
# w-slab: b_A + b_k/b_kb/b_gk/b_kg[/b_q/b_qg][BT,BK]  (4.5× or 6.5× BT×BK + BT×BT)
_RECOMPUTE_FWD_U_MEM_MULT = 3.5
_RECOMPUTE_FWD_W_MEM_MULT = 4.5
_RECOMPUTE_FWD_W_MEM_MULT_QG = 6.5
_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 8
_MAX_TILE_FWD = 128
_PREFERRED_TILE = 64


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def _launch_wy_core_grid(kernel, *, task_num: int, kernel_kwargs: dict) -> None:
    num_core = get_npu_properties()["num_aicore"]
    kernel[(num_core,)](task_num=task_num, num_core=num_core, **kernel_kwargs)


def _candidate_tiles(dim: int) -> list[int]:
    cap = min(_MAX_TILE_FWD, triton.next_power_of_2(dim))
    tiles = [b for b in (_PREFERRED_TILE, _MAX_TILE_FWD, 32, 16, 8) if b <= cap]
    return tiles or [_FALLBACK_TILE]


def _get_fwd_tiles(BT: int, K: int, V: int, *, store_qg: bool) -> tuple[int, int]:
    """Minimize V/K slab iterations under independent UB budgets for u- and w-slabs."""
    w_mult = _RECOMPUTE_FWD_W_MEM_MULT_QG if store_qg else _RECOMPUTE_FWD_W_MEM_MULT

    def _max_tile(fixed_dim: int, mem_mult: float) -> int:
        return compute_row_tile_block_size(
            BT, fixed_dim, mem_mult,
            tiling_row=False,
            safety_margin=_SAFETY_MARGIN,
            fallback=_FALLBACK_TILE,
            min_block=8,
            max_block=min(_MAX_TILE_FWD, triton.next_power_of_2(fixed_dim)),
        )

    max_bk = _max_tile(K, w_mult)
    max_bv = _max_tile(V, _RECOMPUTE_FWD_U_MEM_MULT)

    best_cost = None
    best_bk = max(8, min(max_bk, triton.next_power_of_2(K)))
    best_bv = max(8, min(max_bv, triton.next_power_of_2(V)))

    for bk in _candidate_tiles(K):
        if bk > max_bk:
            continue
        for bv in _candidate_tiles(V):
            if bv > max_bv:
                continue
            cost = triton.cdiv(V, bv) + triton.cdiv(K, bk)
            tie_break = bk + bv
            if best_cost is None or cost < best_cost or (cost == best_cost and tie_break > best_bk + best_bv):
                best_cost = cost
                best_bk, best_bv = bk, bv

    return best_bk, best_bv


def _hv_t_npu_arg(x: torch.Tensor, HV: int) -> tuple[torch.Tensor, bool]:
    if HV == 1:
        return x, False
    return x.transpose(1, 2).contiguous(), True


@triton.jit
def _beta_row_ptr(beta_ptr, T, i_t, BT, BETA_T_CONTIG: tl.constexpr, HV: tl.constexpr):
    if BETA_T_CONTIG:
        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        return beta_ptr + o_t
    o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
    return beta_ptr + o_t * HV


@triton.jit
def _gk_tile_ptr(gk_ptr, T, K, i_t, i_k, BT, BK, GK_T_CONTIG: tl.constexpr, HV: tl.constexpr):
    if GK_T_CONTIG:
        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        o_k = i_k * BK + tl.arange(0, BK)
        return gk_ptr + o_t[:, None] * K + o_k[None, :]
    o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    return gk_ptr + o_t[:, None] * (HV * K) + o_k[None, :]


@triton.heuristics({
    "STORE_QG": lambda args: args["qg"] is not None,
    "STORE_KG": lambda args: args["kg"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T", "B", "task_num", "num_core"])
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
    B,
    task_num,
    num_core,
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
    BETA_T_CONTIG: tl.constexpr,
    GK_T_CONTIG: tl.constexpr,
    MASK_DMA: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    T_max = T
    BH = B * HV
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t_o = task_id // BH
        i_bh = task_id % BH
        i_hv = i_bh % HV
        i_h = i_hv // (HV // H)
        is_valid = True
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1,
            ).to(tl.int32)
            if USE_GRAPH:
                # graph padding uses [-1, 0]; exclude it before sequence-dependent memory accesses.
                is_valid = i_n >= 0
        # skip only this task: returning would drop later valid tasks assigned to this core.
        if is_valid:
            if IS_VARLEN:
                bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
                    cu_seqlens + i_n + 1,
                ).to(tl.int64)
                T = (eos - bos).to(tl.int32)
                beta_bh = bos + i_hv * T_max
                gk_bh = i_hv * T_max * K + bos * K
            else:
                i_b = i_bh // HV
                i_t = i_t_o
                bos = tl.cast(i_b, tl.int64) * T
                beta_bh = (tl.cast(i_b, tl.int64) * HV + i_hv) * T_max
                gk_bh = (tl.cast(i_b, tl.int64) * HV + i_hv) * T_max * K

            k_ptr = k + (bos * H + i_h) * K
            v_ptr = v + (bos * HV + i_hv) * V
            u_ptr = u + (bos * HV + i_hv) * V
            w_ptr = w + (bos * HV + i_hv) * K
            A_ptr = A + (bos * HV + i_hv) * BT
            kg_ptr = kg + (bos * HV + i_hv) * K
            if BETA_T_CONTIG:
                beta_ptr = beta + beta_bh
            else:
                beta_ptr = beta + bos * HV + i_hv
            if GK_T_CONTIG:
                gk_ptr = gk + gk_bh
            else:
                gk_ptr = gk + (bos * HV + i_hv) * K
            if STORE_QG:
                q_ptr = q + (bos * H + i_h) * K
                qg_ptr = qg + (bos * HV + i_hv) * K

            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t = (o_t >= 0) & (o_t < T)
            p_b = _beta_row_ptr(beta_ptr, T, i_t, BT, BETA_T_CONTIG, HV)
            b_b = mload(p_b, m_t, MASK_DMA)

            o_i = tl.arange(0, BT)
            m_i = (o_i >= 0) & (o_i < BT)
            m_p_A = m_t[:, None] & m_i[None, :]
            p_A = A_ptr + o_t[:, None] * (HV * BT) + o_i[None, :]

            last_idx = min(tl.cast(i_t, tl.int64) * BT + BT, T) - 1

            for i_v in range(tl.cdiv(V, BV)):
                o_v = i_v * BV + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                m_p_v = m_t[:, None] & m_v[None, :]
                p_v = v_ptr + o_t[:, None] * (HV * V) + o_v[None, :]
                p_u = u_ptr + o_t[:, None] * (HV * V) + o_v[None, :]
                b_v = mload(p_v, m_p_v, MASK_DMA)
                b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
                # Ascend tl.dot may clobber the left operand; reload A each V tile.
                b_A = mload(p_A, m_p_A, MASK_DMA)
                b_u = tl.dot(b_A, b_vb, allow_tf32=False)
                mstore(p_u, b_u.to(p_u.dtype.element_ty), m_p_v, MASK_DMA)

            for i_k in range(tl.cdiv(K, BK)):
                o_k_2 = i_k * BK + tl.arange(0, BK)
                m_k_2 = (o_k_2 >= 0) & (o_k_2 < K)
                m_p_k = m_t[:, None] & m_k_2[None, :]
                p_k = k_ptr + o_t[:, None] * (H * K) + o_k_2[None, :]
                b_k = mload(p_k, m_p_k, MASK_DMA)
                b_kb = b_k * b_b[:, None]

                p_gk = _gk_tile_ptr(gk_ptr, T, K, i_t, i_k, BT, BK, GK_T_CONTIG, HV)
                b_gk = mload(p_gk, m_p_k, MASK_DMA).to(tl.float32)
                b_gk_exp = exp2(b_gk)
                b_kb = b_kb * b_gk_exp

                if STORE_QG:
                    p_q = q_ptr + o_t[:, None] * (H * K) + o_k_2[None, :]
                    p_qg = qg_ptr + o_t[:, None] * (HV * K) + o_k_2[None, :]
                    b_q = mload(p_q, m_p_k, MASK_DMA)
                    mstore(p_qg, (b_q * b_gk_exp).to(p_qg.dtype.element_ty), m_p_k, MASK_DMA)

                if STORE_KG:
                    o_k = i_k * BK + tl.arange(0, BK)
                    m_k = o_k < K
                    if GK_T_CONTIG:
                        b_gn = mload(gk_ptr + tl.cast(last_idx, tl.int64) * K + o_k, m_k, MASK_DMA).to(tl.float32)
                    else:
                        b_gn = mload(gk_ptr + tl.cast(last_idx, tl.int64) * HV * K + o_k, m_k, MASK_DMA).to(tl.float32)
                    b_kg = b_k * exp2(b_gn[None, :] - b_gk)
                    p_kg = kg_ptr + o_t[:, None] * (HV * K) + o_k_2[None, :]
                    mstore(p_kg, b_kg.to(p_kg.dtype.element_ty), m_p_k, MASK_DMA)

                # Ascend tl.dot may clobber the left operand; reload A each K tile.
                b_A = mload(p_A, m_p_A, MASK_DMA)
                b_w = tl.dot(b_A, b_kb.to(b_k.dtype), allow_tf32=False)
                p_w = w_ptr + o_t[:, None] * (HV * K) + o_k_2[None, :]
                mstore(p_w, b_w.to(p_w.dtype.element_ty), m_p_k, MASK_DMA)


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
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BT = A.shape[-1]
    store_qg = q is not None
    BK, BV = _get_fwd_tiles(BT, K, V, store_qg=store_qg)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    beta, beta_t_contig = _hv_t_npu_arg(beta, HV)
    gk, gk_t_contig = _hv_t_npu_arg(gk, HV)

    w = (k.new_zeros if use_graph else k.new_empty)(B, T, HV, K)
    u = (torch.zeros_like if use_graph else torch.empty_like)(v)
    qg = (k.new_zeros if use_graph else k.new_empty)(B, T, HV, K) if store_qg else None
    kg = (k.new_zeros if use_graph else k.new_empty)(B, T, HV, K)

    _launch_wy_core_grid(
        recompute_w_u_fwd_kda_kernel_npu,
        task_num=NT * B * HV,
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
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            BETA_T_CONTIG=beta_t_contig,
            GK_T_CONTIG=gk_t_contig,
            MASK_DMA=need_dma_mask(T, BT, K, BK, V, BV, cu_seqlens is not None),
            USE_GRAPH=use_graph,
        ),
    )
    return w, u, qg, kg
