# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.common.utils import prepare_chunk_offsets


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_G': lambda args: args['g'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['BT', 'BK', 'BV', 'USE_G'],
)
@triton.jit
def chunk_gated_delta_rule_fwd_kernel_h(
    k,
    v,
    d,
    v_new,
    g,
    h,
    h0,
    ht,
    offsets,
    chunk_offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_G: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        if HEAD_FIRST:
            p_h = tl.make_block_ptr(h + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            if HEAD_FIRST:
                b_g_last = tl.load(g + i_nh * T + last_idx)
            else:
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        else:
            b_g_last = None
            last_idx = None
        # since we need to make all DK in the SRAM. we face serve SRAM memory burden. By subchunking we allievate such burden
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            if HEAD_FIRST:
                p_k = tl.make_block_ptr(k + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_d = tl.make_block_ptr(d + i_nh * T*K, (T, K), (K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_v = tl.make_block_ptr(v + i_nh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new+i_nh*T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_g = tl.make_block_ptr(g + i_nh * T, (T,), (1,), (i_t * BT + i_c * BC,), (BC,), (0,)) if USE_G else None
            else:
                p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_d = tl.make_block_ptr(d+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT+i_c*BC, i_v * BV), (BC, BV), (1, 0))
                p_g = tl.make_block_ptr(g+bos*H+i_h, (T,), (H,), (i_t*BT+i_c*BC, ), (BC,), (0,)) if USE_G else None
            b_g = tl.load(p_g, boundary_check=(0, )) if USE_G else None
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(b_g_last - b_g)[None, :]).to(b_k.dtype) if USE_G else b_k
            # [BC, BK]
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_d = (b_d * tl.exp(b_g)[:, None]).to(b_d.dtype) if USE_G else b_d
            # [BC, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_v2 = b_v - tl.dot(b_d, b_h.to(b_d.dtype))
            # [BK, BV]
            tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
            b_hc += tl.dot(b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_h *= tl.exp(b_g_last) if USE_G else 1
        b_h += b_hc

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_G': lambda args: args['g'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['BT', 'BK', 'BV', 'USE_G'],
)
@triton.jit
def chunk_gated_delta_rule_bwd_kernel_dhu(
    q,
    k,
    d,
    g,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    offsets,
    chunk_offsets,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_G: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        if HEAD_FIRST:
            p_dh = tl.make_block_ptr(dh + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_dh = tl.make_block_ptr(dh + ((boh+i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            if HEAD_FIRST:
                bg_last = tl.load(g + i_nh * T + last_idx)
            else:
                bg_last = tl.load(g + (bos + last_idx) * H + i_h)
        else:
            bg_last = None
            last_idx = None
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            if HEAD_FIRST:
                p_q = tl.make_block_ptr(q + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_k = tl.make_block_ptr(k + i_nh * T*K, (T, K), (K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_d = tl.make_block_ptr(d + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_dv = tl.make_block_ptr(dv + i_nh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_do = tl.make_block_ptr(do + i_nh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_g = tl.make_block_ptr(g + i_nh * T, (T,), (1,), (i_t * BT + i_c * BC,), (BC,), (0,)) if USE_G else None
                p_dv2 = tl.make_block_ptr(dv2 + i_nh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            else:
                p_q = tl.make_block_ptr(q+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_d = tl.make_block_ptr(d+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_dv = tl.make_block_ptr(dv+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_do = tl.make_block_ptr(do+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_g = tl.make_block_ptr(g+bos*H+i_h, (T,), (H,), (i_t*BT + i_c * BC,), (BC,), (0,)) if USE_G else None
                p_dv2 = tl.make_block_ptr(dv2+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_g = tl.load(p_g, boundary_check=(0,)) if USE_G else None
            # [BK, BT]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale * tl.exp(b_g)[None, :]).to(b_q.dtype) if USE_G else (b_q * scale).to(b_q.dtype)
            # [BT, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(bg_last - b_g)[:, None]).to(b_k.dtype) if USE_G else b_k
            b_d = (b_d * tl.exp(b_g)[None, :]).to(b_d.dtype) if USE_G else b_d
            # [BT, V]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dv2 = b_dv + tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            tl.store(p_dv2, b_dv2.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
            # [BK, BV]
            b_dh_tmp += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)
            b_dh_tmp -= tl.dot(b_d, b_dv2.to(b_q.dtype), allow_tf32=False)
        b_dh *= tl.exp(bg_last) if USE_G else 1
        b_dh += b_dh_tmp

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, u.shape[-1]
    else:
        B, T, H, K, V = *k.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_chunk_offsets(offsets, BT)
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    # A100
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64
    else:
        BV = 32
        BC = 64 if K <= 128 else 32
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    if head_first:
        h = k.new_empty(B, H, NT, K, V)
    else:
        h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u)
    grid = (NK, NV, N * H)

    chunk_gated_delta_rule_fwd_kernel_h[grid](
        k=k,
        v=u,
        d=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        NT=NT,
        HEAD_FIRST=head_first
    )
    return h, v_new, final_state


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *q.shape, do.shape[-1]
    else:
        B, T, H, K, V = *q.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_chunk_offsets(offsets, BT)
        NT = chunk_offsets[-1]

    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension being larger than 256."
    # H100
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    # A100
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64 if K <= 128 else 32
    else:
        BV = 32
        BC = 64 if K <= 128 else 32
    BC = min(BT, BC)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    if head_first:
        dh = q.new_empty(B, H, NT, K, V)
    else:
        dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    grid = (NK, NV, N * H)
    chunk_gated_delta_rule_bwd_kernel_dhu[grid](
        q=q,
        k=k,
        d=w,
        g=g,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return dh, dh0, dv2
