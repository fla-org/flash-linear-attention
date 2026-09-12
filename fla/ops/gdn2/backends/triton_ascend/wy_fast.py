# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN-2 WY-representation kernels for Triton-Ascend."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size, get_npu_properties

# share A between both stages, while the channel-wise gates keep the V and K
# slabs independent. The multipliers conservatively include the extra gate tile.
_RECOMPUTE_FWD_U_MEM_MULT = 4.5
_RECOMPUTE_FWD_W_MEM_MULT = 5.5
_RECOMPUTE_FWD_W_MEM_MULT_QG = 7.5
_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 16
_MAX_TILE_FWD = 128
_PREFERRED_TILE = 64
_MAX_K = 256


def _launch_wy_core_grid(kernel, *, task_num: int, kernel_kwargs: dict) -> None:
    num_core = get_npu_properties()["num_aicore"]
    kernel[(num_core,)](
        task_num=task_num,
        num_core=num_core,
        **ascend_compile_kwargs(),
        **kernel_kwargs,
    )


def _candidate_tiles(dim: int) -> list[int]:
    cap = min(_MAX_TILE_FWD, max(16, triton.next_power_of_2(dim)))
    tiles = [tile for tile in (_PREFERRED_TILE, _MAX_TILE_FWD, 32, 16) if tile <= cap]
    return tiles or [_FALLBACK_TILE]


def _get_fwd_tiles(BT: int, K: int, V: int, *, store_qg: bool) -> tuple[int, int]:
    """Choose independent K/V slabs within their respective peak UB budgets."""
    w_mult = _RECOMPUTE_FWD_W_MEM_MULT_QG if store_qg else _RECOMPUTE_FWD_W_MEM_MULT

    def _max_tile(dim: int, mem_mult: float) -> int:
        return compute_row_tile_block_size(
            BT,
            dim,
            mem_mult,
            tiling_row=False,
            safety_margin=_SAFETY_MARGIN,
            fallback=_FALLBACK_TILE,
            min_block=_FALLBACK_TILE,
            max_block=min(_MAX_TILE_FWD, max(16, triton.next_power_of_2(dim))),
        )

    max_bk = _max_tile(K, w_mult)
    max_bv = _max_tile(V, _RECOMPUTE_FWD_U_MEM_MULT)
    best_cost = None
    best_bk = max(_FALLBACK_TILE, min(max_bk, triton.next_power_of_2(K)))
    best_bv = max(_FALLBACK_TILE, min(max_bv, triton.next_power_of_2(V)))

    for bk in _candidate_tiles(K):
        if bk > max_bk:
            continue
        for bv in _candidate_tiles(V):
            if bv > max_bv:
                continue
            cost = triton.cdiv(K, bk) + triton.cdiv(V, bv)
            tie_break = bk + bv
            if best_cost is None or cost < best_cost or (cost == best_cost and tie_break > best_bk + best_bv):
                best_cost = cost
                best_bk, best_bv = bk, bv

    return best_bk, best_bv


@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'B', 'task_num', 'num_core'])
def recompute_w_u_fwd_gdn2_kernel_npu(
    q,
    k,
    qg,
    kg,
    v,
    b,
    w_gate,
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
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    T_seq = T
    BH = B * H
    core_id = tl.program_id(0)

    for tid in tl.range(core_id, task_num, num_core):
        task_id = tl.cast(tid, tl.int32)
        i_t_o = task_id // BH
        i_bh = task_id % BH
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_t_o64 = tl.cast(i_t_o, tl.int64)
            i_n = tl.load(chunk_indices + i_t_o64 * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_t_o64 * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_cur = (eos - bos).to(tl.int32)
        else:
            i_t = i_t_o
            bos = tl.cast(i_b, tl.int64) * T_seq
            T_cur = T_seq

        i_h64 = tl.cast(i_h, tl.int64)
        i_t64 = tl.cast(i_t, tl.int64)
        o_t = tl.arange(0, BT)
        o_t64 = tl.cast(o_t, tl.int64)
        o_A64 = tl.cast(tl.arange(0, BT), tl.int64)
        local_t64 = i_t64 * BT + o_t64
        m_t = local_t64 < T_cur

        head_A = (bos * H + i_h64) * BT
        p_A = A + head_A + local_t64[:, None] * (H * BT) + o_A64[None, :]

        for i_v in range(tl.cdiv(V, BV)):
            o_v64 = tl.cast(i_v * BV + tl.arange(0, BV), tl.int64)
            m_v = m_t[:, None] & (o_v64[None, :] < V)
            head_v = (bos * H + i_h64) * V
            p_v = v + head_v + local_t64[:, None] * (H * V) + o_v64[None, :]
            p_wg = w_gate + head_v + local_t64[:, None] * (H * V) + o_v64[None, :]
            b_v = tl.load(p_v, mask=m_v, other=0.0)
            b_wg = tl.load(p_wg, mask=m_v, other=0.0)
            b_vb = (b_v * b_wg).to(b_v.dtype)
            # reload A for each tile because Ascend tl.dot may clobber the left operand.
            b_A = tl.load(p_A, mask=m_t[:, None], other=0.0)
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)
            p_u = u + head_v + local_t64[:, None] * (H * V) + o_v64[None, :]
            tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

        for i_k in range(tl.cdiv(K, BK)):
            o_k64 = tl.cast(i_k * BK + tl.arange(0, BK), tl.int64)
            m_k = m_t[:, None] & (o_k64[None, :] < K)
            head_k = (bos * H + i_h64) * K
            p_k = k + head_k + local_t64[:, None] * (H * K) + o_k64[None, :]
            p_b = b + head_k + local_t64[:, None] * (H * K) + o_k64[None, :]
            p_gk = gk + head_k + local_t64[:, None] * (H * K) + o_k64[None, :]
            b_k = tl.load(p_k, mask=m_k, other=0.0)
            b_b = tl.load(p_b, mask=m_k, other=0.0)
            b_gk = tl.load(p_gk, mask=m_k, other=0.0).to(tl.float32)
            b_gk_exp = exp2(b_gk)
            b_kb = b_k * b_b
            b_kb *= b_gk_exp

            if STORE_QG:
                p_q = q + head_k + local_t64[:, None] * (H * K) + o_k64[None, :]
                p_qg = qg + head_k + local_t64[:, None] * (H * K) + o_k64[None, :]
                b_q = tl.load(p_q, mask=m_k, other=0.0)
                tl.store(p_qg, (b_q * b_gk_exp).to(p_qg.dtype.element_ty), mask=m_k)

            last_idx64 = min(i_t64 * BT + BT, T_cur) - 1
            m_last_k = o_k64 < K
            p_gn = gk + ((bos + last_idx64) * H + i_h64) * K + o_k64
            b_gn = tl.load(p_gn, mask=m_last_k, other=0.0).to(tl.float32)
            b_kg = b_k * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_gk), 0.0)
            p_kg = kg + head_k + local_t64[:, None] * (H * K) + o_k64[None, :]
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), mask=m_k)

            # reload A after every prior dot because it is reused as the next lhs.
            b_A = tl.load(p_A, mask=m_t[:, None], other=0.0)
            b_w = tl.dot(b_A, b_kb.to(b_k.dtype), allow_tf32=False)
            p_w = w + head_k + local_t64[:, None] * (H * K) + o_k64[None, :]
            tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)


@input_guard
def recompute_w_u_fwd_gdn2_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    if BT != 64:
        raise ValueError(f'GDN-2 Ascend WY requires chunk_size=64, got {BT}')
    if not 1 <= K <= _MAX_K:
        raise ValueError(f'GDN-2 Ascend WY requires 1 <= K <= {_MAX_K}, got K={K}')
    if V < 1:
        raise ValueError(f'GDN-2 Ascend WY requires V >= 1, got V={V}')
    if gk is None:
        raise ValueError('GDN-2 Ascend WY requires gk')
    if cu_seqlens is not None and B != 1:
        raise ValueError('GDN-2 Ascend WY requires batch size 1 for packed sequences')
    BK, BV = _get_fwd_tiles(BT, K, V, store_qg=q is not None)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    qg = torch.empty_like(q) if q is not None else None
    kg = torch.empty_like(k)
    _launch_wy_core_grid(
        recompute_w_u_fwd_gdn2_kernel_npu,
        task_num=NT * B * H,
        kernel_kwargs=dict(
            q=q,
            k=k,
            qg=qg,
            kg=kg,
            v=v,
            b=b,
            w_gate=w_gate,
            w=w,
            u=u,
            A=A,
            gk=gk,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            B=B,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
        ),
    )
    return w, u, qg, kg
