# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN-2 WY-representation kernels for Triton-Ascend."""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.gdn2.wy_fast import recompute_w_u_fwd_gdn2
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.ops.utils.triton_ascend.addressing import chunk_token_rows
from fla.utils import ascend_compile_kwargs, get_multiprocessor_count, input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size, get_ub_manager


@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_gdn2_fp32_kernel_npu(
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
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    row_base, rows, m_rows = chunk_token_rows(i_t, T, BT)
    o_A = tl.arange(0, BT)
    m_A = m_rows[:, None] & (o_A[None, :] < BT)
    p_A = A + (bos * H + i_h) * BT + row_base * (H * BT) + rows[:, None] * (H * BT) + o_A[None, :]

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_rows[:, None] & (o_v[None, :] < V)
        p_v = v + (bos * H + i_h) * V + row_base * (H * V) + rows[:, None] * (H * V) + o_v[None, :]
        p_u = u + (bos * H + i_h) * V + row_base * (H * V) + rows[:, None] * (H * V) + o_v[None, :]
        p_wg = w_gate + (bos * H + i_h) * V + row_base * (H * V) + rows[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)
        b_wg = tl.load(p_wg, mask=m_v, other=0.0).to(tl.float32)
        b_vb = b_v * b_wg
        b_A = tl.load(p_A, mask=m_A, other=0.0).to(tl.float32)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_kk = m_rows[:, None] & (o_k[None, :] < K)
        p_w = w + (bos * H + i_h) * K + row_base * (H * K) + rows[:, None] * (H * K) + o_k[None, :]
        p_k = k + (bos * H + i_h) * K + row_base * (H * K) + rows[:, None] * (H * K) + o_k[None, :]
        p_b = b + (bos * H + i_h) * K + row_base * (H * K) + rows[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kk, other=0.0).to(tl.float32)
        b_b = tl.load(p_b, mask=m_kk, other=0.0).to(tl.float32)
        b_kb = b_k * b_b

        p_gk = gk + (bos * H + i_h) * K + row_base * (H * K) + rows[:, None] * (H * K) + o_k[None, :]
        b_gk = tl.load(p_gk, mask=m_kk, other=0.0).to(tl.float32)
        b_kb *= exp2(b_gk)

        if STORE_QG:
            p_q = q + (bos * H + i_h) * K + row_base * (H * K) + rows[:, None] * (H * K) + o_k[None, :]
            p_qg = qg + (bos * H + i_h) * K + row_base * (H * K) + rows[:, None] * (H * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_kk, other=0.0)
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), mask=m_kk)

        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k,
                mask=m_k, other=0.,
            ).to(tl.float32)
            b_kg = b_k * tl.where(
                m_rows[:, None],
                exp2(b_gn[None, :] - b_gk),
                0,
            )
            p_kg = kg + (bos * H + i_h) * K + row_base * (H * K) + rows[:, None] * (H * K) + o_k[None, :]
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), mask=m_kk)

        b_A = tl.load(p_A, mask=m_A, other=0.0).to(tl.float32)
        b_w = tl.dot(b_A, b_kb.to(tl.float32), allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_kk)


def _recompute_w_u_fwd_gdn2_fp32_npu(
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
    """NPU WY recompute. ``A @ x`` and the returned factors stay in fp32.

    The inter-chunk state is allocated from ``k``'s dtype. Writing ``w`` / ``u`` /
    ``kg`` back to bf16 rounds that state on every chunk and pushes a long
    sequence past the bf16 reference tolerance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = BV = 64
    if (
        cu_seqlens is None and BT == 64 and T % BT == 0
        and K in (32, 64, 128) and V in (32, 64, 128)
    ):
        BK, BV = _get_fwd_tiles(BT, K, V, store_qg=q is not None, ub_capacity_bits=get_ub_manager().ub_capacity_bits)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty(B, T, H, K, device=k.device, dtype=torch.float32)
    u = torch.empty(B, T, H, V, device=v.device, dtype=torch.float32)
    qg = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32) if q is not None else None
    kg = torch.empty(B, T, H, K, device=k.device, dtype=torch.float32) if gk is not None else None
    recompute_w_u_fwd_gdn2_fp32_kernel_npu[(NT, B * H)](
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
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u, qg, kg


# share A between both stages, while the channel-wise gates keep the V and K
# slabs independent. The multipliers conservatively include the extra gate tile.
_RECOMPUTE_FWD_U_MEM_MULT = 4.5
_RECOMPUTE_FWD_W_MEM_MULT = 4.5
_RECOMPUTE_FWD_W_MEM_MULT_QG = 6.5
_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 16
_MAX_TILE_FWD = 128
_PREFERRED_TILE = 64
_MAX_K = 256


def _launch_wy_core_grid(kernel, *, task_num: int, device_index: int, kernel_kwargs: dict) -> None:
    num_core = get_multiprocessor_count(device_index, use_aicore=True)
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


@lru_cache(maxsize=128)
def _get_fwd_tiles(BT: int, K: int, V: int, *, store_qg: bool, ub_capacity_bits: int) -> tuple[int, int]:
    """Choose K/V slabs, caching plans separately for each detected UB capacity."""
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
    'IS_ALIGNED': lambda args: args['cu_seqlens'] is None and args['T'] % args['BT'] == 0,
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
    IS_ALIGNED: tl.constexpr,
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
        m_t = tl.full((BT,), True, tl.int1) if IS_ALIGNED else local_t64 < T_cur

        head_A = (bos * H + i_h64) * BT
        p_A = A + head_A + local_t64[:, None] * (H * BT) + o_A64[None, :]

        for i_v in range(tl.cdiv(V, BV)):
            o_v64 = tl.cast(i_v, tl.int64) * BV + tl.cast(tl.arange(0, BV), tl.int64)
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
            o_k64 = tl.cast(i_k, tl.int64) * BK + tl.cast(tl.arange(0, BK), tl.int64)
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

            last_idx64 = (i_t64 + 1) * BT - 1 if IS_ALIGNED else min(i_t64 * BT + BT, T_cur) - 1
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


def _requires_fp32_wy(A: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor | None) -> bool:
    return A.dtype != k.dtype or A.dtype != v.dtype or (q is not None and q.dtype != A.dtype)


def recompute_w_u_fwd_gdn2_verifier(
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[bool, str | None]:
    if gk is None:
        return False, "GDN-2 Ascend WY requires gk"
    float_tensors = (k, v, b, w_gate, A, gk, *(t for t in (q,) if t is not None))
    if any(t.ndim != 4 for t in float_tensors):
        return False, "GDN-2 Ascend WY requires rank-4 tensors"
    k_shape, k_device = k.shape, k.device
    requires_fp32 = _requires_fp32_wy(A, k, v, q)
    if not requires_fp32 and A.shape[-1] != 64:
        return False, f"GDN-2 Ascend WY requires chunk_size=64, got {A.shape[-1]}"
    if A.shape[:3] != k_shape[:3]:
        return False, "GDN-2 Ascend WY requires A to match k on [B, T, H]"
    if any(t.shape != k_shape for t in (b, gk, *(t for t in (q,) if t is not None))):
        return False, "GDN-2 Ascend WY requires b/gk/q shapes to match k"
    if v.shape[:3] != k_shape[:3] or w_gate.shape != v.shape:
        return False, "GDN-2 Ascend WY requires v/w_gate to match k on [B, T, H]"
    if cu_seqlens is not None and k_shape[0] != 1:
        return False, "GDN-2 Ascend WY requires batch size 1 for packed sequences"
    if cu_seqlens is not None and cu_seqlens.ndim != 1:
        return False, "GDN-2 Ascend WY requires cu_seqlens to be rank 1"
    if chunk_indices is not None and (chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2):
        return False, "GDN-2 Ascend WY requires chunk_indices shape [NT, 2]"
    K = int(k_shape[-1])
    if K < 1 or (not requires_fp32 and K > _MAX_K):
        return False, f"GDN-2 Ascend WY requires 1 <= K <= 256, got K={K}"
    V = int(v.shape[-1])
    if V < 1:
        return False, f"GDN-2 Ascend WY requires V >= 1, got V={V}"
    tensors = (*float_tensors, *(t for t in (cu_seqlens, chunk_indices) if t is not None))
    if any(d.type != "npu" or d != k_device for d in (t.device for t in tensors)):
        return False, "GDN-2 Ascend WY requires tensors on the same NPU device"
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if any(t.dtype not in supported for t in float_tensors):
        return False, "GDN-2 Ascend WY received an unsupported dtype"
    index_tensors = tuple(t for t in (cu_seqlens, chunk_indices) if t is not None)
    if any(t.dtype not in (torch.int32, torch.int64) for t in index_tensors):
        return False, "GDN-2 Ascend WY requires int32 or int64 sequence indices"
    return True, None


@register(recompute_w_u_fwd_gdn2, backend=TritonAscendBackend, verifier=recompute_w_u_fwd_gdn2_verifier)
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
    if gk is None:
        raise ValueError('GDN-2 Ascend WY requires gk')
    if _requires_fp32_wy(A, k, v, q):
        return _recompute_w_u_fwd_gdn2_fp32_npu(
            k=k,
            v=v,
            b=b,
            w_gate=w_gate,
            A=A,
            q=q,
            gk=gk,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    if BT != 64:
        raise ValueError(f'GDN-2 Ascend WY requires chunk_size=64, got {BT}')
    if not 1 <= K <= _MAX_K:
        raise ValueError(f'GDN-2 Ascend WY requires 1 <= K <= {_MAX_K}, got K={K}')
    if V < 1:
        raise ValueError(f'GDN-2 Ascend WY requires V >= 1, got V={V}')
    if cu_seqlens is not None and B != 1:
        raise ValueError('GDN-2 Ascend WY requires batch size 1 for packed sequences')
    BK, BV = _get_fwd_tiles(BT, K, V, store_qg=q is not None, ub_capacity_bits=get_ub_manager().ub_capacity_bits)

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
        device_index=k.device.index,
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
