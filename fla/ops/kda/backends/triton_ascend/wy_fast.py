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

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size

# recompute_w_u_fwd (fused u/w loop): peak UB is max(u-slab, w-slab), not sum — same as unfused
_RECOMPUTE_FWD_FUSED_MEM_MULT = 6.0
_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 8
_MAX_TILE_FWD = 64


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def _launch_wy_core_grid(kernel, *, task_num: int, kernel_kwargs: dict) -> None:
    num_core = get_npu_properties()["num_aicore"]
    kernel[(num_core,)](task_num=task_num, num_core=num_core, **kernel_kwargs)


def _get_fwd_tiles(BT: int, K: int, V: int) -> tuple[int, int]:
    """Unified tile for fused u/w loop; both slabs share the same UB budget."""
    max_k = max(K, V)
    max_block = min(_MAX_TILE_FWD, triton.next_power_of_2(max_k))
    b_tile = compute_row_tile_block_size(
        BT, max_k, _RECOMPUTE_FWD_FUSED_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=max_block,
    )
    BK = min(b_tile, triton.next_power_of_2(K))
    BV = min(b_tile, triton.next_power_of_2(V))
    return max(8, BK), max(8, BV)


def _hv_t_npu_arg(x: torch.Tensor, HV: int) -> tuple[torch.Tensor, bool]:
    if HV == 1:
        return x, False
    return x.transpose(1, 2).contiguous(), True


@triton.jit
def _beta_block_ptr(beta_ptr, T, i_t, BT, BETA_T_CONTIG: tl.constexpr, HV: tl.constexpr):
    if BETA_T_CONTIG:
        return tl.make_block_ptr(beta_ptr, (T,), (1,), (i_t * BT,), (BT,), (0,))
    return tl.make_block_ptr(beta_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))


@triton.jit
def _gk_block_ptr(gk_ptr, T, K, i_t, i_k, BT, BK, GK_T_CONTIG: tl.constexpr, HV: tl.constexpr):
    if GK_T_CONTIG:
        return tl.make_block_ptr(gk_ptr, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    return tl.make_block_ptr(gk_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))


@triton.jit
def _recompute_w_slab(
    i_uv,
    w_ptr,
    k_ptr,
    gk_ptr,
    kg_ptr,
    q,
    qg,
    b_A,
    b_b,
    bos,
    i_h,
    i_hv,
    T,
    K,
    H,
    HV,
    i_t,
    BT,
    BK,
    last_idx,
    STORE_QG: tl.constexpr,
    GK_T_CONTIG: tl.constexpr,
):
    p_w = tl.make_block_ptr(
        w_ptr, (T, K), (HV * K, 1), (i_t * BT, i_uv * BK), (BT, BK), (1, 0),
    )
    p_k = tl.make_block_ptr(
        k_ptr, (T, K), (H * K, 1), (i_t * BT, i_uv * BK), (BT, BK), (1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_kb = b_k * b_b[:, None]

    p_gk = _gk_block_ptr(gk_ptr, T, K, i_t, i_uv, BT, BK, GK_T_CONTIG, HV)
    b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
    b_kb = b_kb * exp2(b_gk)

    if STORE_QG:
        q_ptr = q + (bos * H + i_h) * K
        qg_ptr = qg + (bos * HV + i_hv) * K
        p_q = tl.make_block_ptr(
            q_ptr, (T, K), (H * K, 1), (i_t * BT, i_uv * BK), (BT, BK), (1, 0),
        )
        p_qg = tl.make_block_ptr(
            qg_ptr, (T, K), (HV * K, 1), (i_t * BT, i_uv * BK), (BT, BK), (1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * exp2(b_gk)
        tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))

    o_k = i_uv * BK + tl.arange(0, BK)
    m_k = o_k < K
    if GK_T_CONTIG:
        b_gn = tl.load(gk_ptr + last_idx * K + o_k, mask=m_k, other=0.).to(tl.float32)
    else:
        b_gn = tl.load(gk_ptr + last_idx * HV * K + o_k, mask=m_k, other=0.).to(tl.float32)
    b_kg = b_k * tl.where(
        (i_t * BT + tl.arange(0, BT) < T)[:, None],
        exp2(b_gn[None, :] - b_gk),
        0,
    )
    p_kg = tl.make_block_ptr(
        kg_ptr, (T, K), (HV * K, 1), (i_t * BT, i_uv * BK), (BT, BK), (1, 0),
    )
    tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

    b_w = tl.dot(b_A, b_kb.to(tl.float32), allow_tf32=False)
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'K_EQ_V': lambda args: args['K'] == args['V'],
})
@triton.jit(do_not_specialize=['T', 'B', 'task_num', 'num_core'])
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
    IS_VARLEN: tl.constexpr,
    BETA_T_CONTIG: tl.constexpr,
    GK_T_CONTIG: tl.constexpr,
    K_EQ_V: tl.constexpr,
):
    T_max = T
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t_o = task_id // (B * HV)
        i_bh = task_id % (B * HV)
        i_hv = i_bh % HV
        i_h = i_hv // (HV // H)
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1,
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1,
            ).to(tl.int32)
            T = eos - bos
            # [HV, T, K] layout: head hv token t key k -> hv * T_max * K + (bos + t) * K
            beta_bh = bos + i_hv * T_max
            gk_bh = i_hv * T_max * K + bos * K
        else:
            i_b = i_bh // HV
            i_t = i_t_o
            bos = i_b * T
            beta_bh = i_b * HV * T_max + i_hv * T_max
            gk_bh = (i_b * HV + i_hv) * T_max * K

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
        p_b = _beta_block_ptr(beta_ptr, T, i_t, BT, BETA_T_CONTIG, HV)
        b_b = tl.load(p_b, boundary_check=(0,))

        p_A = tl.make_block_ptr(A_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)

        last_idx = min(i_t * BT + BT, T) - 1
        if K_EQ_V:
            for i_uv in range(tl.cdiv(V, BV)):
                p_v = tl.make_block_ptr(
                    v_ptr, (T, V), (HV * V, 1), (i_t * BT, i_uv * BV), (BT, BV), (1, 0),
                )
                p_u = tl.make_block_ptr(
                    u_ptr, (T, V), (HV * V, 1), (i_t * BT, i_uv * BV), (BT, BV), (1, 0),
                )
                b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
                b_vb = b_v * b_b[:, None]
                b_u = tl.dot(b_A, b_vb, allow_tf32=False)
                tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

                _recompute_w_slab(
                    i_uv, w_ptr, k_ptr, gk_ptr, kg_ptr, q, qg,
                    b_A, b_b, bos, i_h, i_hv, T, K, H, HV, i_t, BT, BK, last_idx,
                    STORE_QG=STORE_QG, GK_T_CONTIG=GK_T_CONTIG,
                )
        else:
            n_v = tl.cdiv(V, BV)
            n_k = tl.cdiv(K, BK)
            for i_uv in range(tl.maximum(n_v, n_k)):
                if i_uv < n_v:
                    p_v = tl.make_block_ptr(
                        v_ptr, (T, V), (HV * V, 1), (i_t * BT, i_uv * BV), (BT, BV), (1, 0),
                    )
                    p_u = tl.make_block_ptr(
                        u_ptr, (T, V), (HV * V, 1), (i_t * BT, i_uv * BV), (BT, BV), (1, 0),
                    )
                    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
                    b_vb = b_v * b_b[:, None]
                    b_u = tl.dot(b_A, b_vb, allow_tf32=False)
                    tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

                if i_uv < n_k:
                    _recompute_w_slab(
                        i_uv, w_ptr, k_ptr, gk_ptr, kg_ptr, q, qg,
                        b_A, b_b, bos, i_h, i_hv, T, K, H, HV, i_t, BT, BK, last_idx,
                        STORE_QG=STORE_QG, GK_T_CONTIG=GK_T_CONTIG,
                    )


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

    beta, beta_t_contig = _hv_t_npu_arg(beta, HV)
    gk, gk_t_contig = _hv_t_npu_arg(gk, HV)

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    qg = k.new_empty(B, T, HV, K) if store_qg else None
    kg = k.new_empty(B, T, HV, K)

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
            STORE_QG=store_qg,
            IS_VARLEN=is_varlen,
            BETA_T_CONTIG=beta_t_contig,
            GK_T_CONTIG=gk_t_contig,
        ),
    )
    return w, u, qg, kg
