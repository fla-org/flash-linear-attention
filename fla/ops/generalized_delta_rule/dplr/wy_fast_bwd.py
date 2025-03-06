
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.utils import device_capacity, is_intel_a770, use_cuda_graph

triton_config = {'grf_mode': 'large'} if is_intel_a770 else {}


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config(triton_config, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def bwd_prepare_wy_repr_kernel(
    A_ab_inv,
    A_ak,
    ag,
    v,
    dw,
    du,
    dv,
    dv0,
    dag,
    dAak,
    dAab,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_Aab_inv_t = tl.make_block_ptr(A_ab_inv + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
        p_Aak_t = tl.make_block_ptr(A_ak + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
        p_dAak = tl.make_block_ptr(dAak + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_dAab = tl.make_block_ptr(dAab + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_Aak_t = tl.make_block_ptr(A_ak + (bos*H + i_h) * BT,  (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))
        p_Aab_inv_t = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))
        p_dAak = tl.make_block_ptr(dAak + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_dAab = tl.make_block_ptr(dAab + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    b_A_ab_inv_t = tl.load(p_Aab_inv_t, boundary_check=(0, 1))
    b_A_ak_t = tl.load(p_Aak_t, boundary_check=(0, 1))
    b_A_ak_t = tl.where(tl.arange(0, BT)[:, None] < tl.arange(0, BT)[None, :], b_A_ak_t, 0)
    b_A_ab_inv_t = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A_ab_inv_t, 0)
    b_A_tmp_t = tl.dot(b_A_ak_t, b_A_ab_inv_t).to(v.dtype.element_ty)
    b_dA_tmp = tl.zeros([BT, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv0 = tl.make_block_ptr(dv0 + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_du = tl.make_block_ptr(du + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv0 = tl.make_block_ptr(dv0 + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_du = tl.make_block_ptr(du + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA_tmp += tl.dot(b_du.to(b_v.dtype), tl.trans(b_v))
        b_dv0 = tl.load(p_dv0, boundary_check=(0, 1))
        b_dv = b_dv0 + tl.dot(b_A_tmp_t, b_du)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_dA_tmp = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dA_tmp, 0)
    b_dA_ak = tl.dot(b_A_ab_inv_t, b_dA_tmp)
    b_dA_ak = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dA_ak, 0)
    tl.store(p_dAak, b_dA_ak, boundary_check=(0, 1))
    b_dA_ab_inv = tl.dot(b_dA_tmp, b_A_ak_t)

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_ag = tl.make_block_ptr(ag + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dag = tl.make_block_ptr(dag + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_ag = tl.make_block_ptr(ag + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dag = tl.make_block_ptr(dag + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_ag = tl.load(p_ag, boundary_check=(0, 1))
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA_ab_inv += tl.dot(b_dw, tl.trans(b_ag))
        b_dag = tl.dot(b_A_ab_inv_t.to(b_dw.dtype), b_dw)
        tl.store(p_dag, b_dag.to(p_dag.dtype.element_ty), boundary_check=(0, 1))

    # if we know dL/dA^(-1), for dL/dA, we can use the following formula:
    # dL/dA = -(A^(-1))^T @ (dL/dA^(-1)) @ (A^(-1))^T
    # in the fwd pass we use fwd substitution to calculate (I-lower(A_ab))^-1.
    # denote A = I - lower(A_ab), B = A^-1
    # in the backward pass.
    # dL/dA = -(B)^T @ (dL/dB) @ B^T
    # dL/dA_ab = lower(B^T @ dL/dB @ B^T)
    b_dA_ab_inv = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dA_ab_inv, 0)
    b_dA_ab_inv = tl.dot(b_A_ab_inv_t, b_dA_ab_inv)
    b_dA_ab_inv = tl.dot(b_dA_ab_inv, b_A_ab_inv_t)
    b_dA_ab_inv = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dA_ab_inv, 0)
    tl.store(p_dAab, b_dA_ab_inv, boundary_check=(0, 1))


def chunk_dplr_bwd_wy(
    A_ab_inv: torch.Tensor,
    A_ak: torch.Tensor,
    v: torch.Tensor,
    ag: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dv0: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A_ab_inv, A_ak, v, ag, dw, du = map(lambda x: x.contiguous(), [A_ab_inv, A_ak, v, ag, dw, du])
    if head_first:
        B, H, T, K, V = *dw.shape, du.shape[-1]
    else:
        B, T, H, K, V = *dw.shape, du.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        NT = len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64) if device_capacity else min(triton.next_power_of_2(V), 32)

    dA_ab = torch.empty_like(A_ab_inv, dtype=torch.float)
    dA_ak = torch.empty_like(A_ak, dtype=torch.float)
    dv = torch.empty_like(v)
    dag = torch.empty_like(ag)

    bwd_prepare_wy_repr_kernel[(NT, B * H)](
        A_ab_inv=A_ab_inv,
        A_ak=A_ak,
        ag=ag,
        v=v,
        dw=dw,
        du=du,
        dv=dv,
        dv0=dv0,
        dag=dag,
        dAak=dA_ak,
        dAab=dA_ab,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return dA_ab, dA_ak, dv, dag
