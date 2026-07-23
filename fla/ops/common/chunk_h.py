# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs, check_shared_mem

BKV_LIST = [32, 64] if check_shared_mem() else [16, 32]


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'STORE_MMA_STATE': lambda args: args['h_mma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h(
    k,
    v,
    h,
    h_mma,
    g,
    g_gamma,
    gk,
    gv,
    h0,
    ht,
    cu_seqlens,
    split_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_MMA_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = i_n * NS
    NTS = BS // BT

    if USE_G_GAMMA:
        # decay rate given the head index
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # [BK, BV] accumulator; STATE_V_FIRST only flips the stored state's HBM layout to [V, K], applied at the load/store below.
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            b_h = tl.trans(tl.load(p_h0, boundary_check=(0, 1))).to(tl.float32)
        else:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        i_s = i_t // NTS
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(h + o_h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            if STORE_MMA_STATE:
                p_h_mma = tl.make_block_ptr(h_mma + o_h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            if STORE_MMA_STATE:
                p_h_mma = tl.make_block_ptr(h_mma + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        if i_t % NTS == 0:
            tl.store(p_h, (tl.trans(b_h) if STATE_V_FIRST else b_h).to(p_h.dtype.element_ty), boundary_check=(0, 1))
            if STORE_MMA_STATE:
                tl.store(
                    p_h_mma,
                    (tl.trans(b_h) if STATE_V_FIRST else b_h).to(p_h_mma.dtype.element_ty),
                    boundary_check=(0, 1),
                )
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1

        # scalar decay
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_h *= exp2(b_g_last)
            b_v = (b_v * exp2(b_g_last - b_g)[:, None]).to(b_v.dtype)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_h *= exp2(b_g_last)
            b_v = (b_v * exp2(b_g_last - b_g)[:, None]).to(b_v.dtype)

        # vector decay, h = Diag(gk) @ h
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_h *= exp2(b_gk_last)[:, None]
            b_k = (b_k * exp2(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

        # vector decay, h = h @ Diag(gv)
        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_h *= exp2(b_gv_last)[None, :]
            b_v = (b_v * exp2(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

        b_h += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(ht + i_nh * K*V, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            tl.store(p_ht, tl.trans(b_h).to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        else:
            p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'STORE_MMA_STATE': lambda args: args['dh_mma'] is not None,
    'FUSE_HDH_LAST': lambda args: args['h_for_hdh'] is not None and args['hdh_last'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dh(
    q,
    g,
    g_gamma,
    gk,
    gv,
    do,
    dh,
    dh_mma,
    h_for_hdh,
    hdh_last,
    dht,
    dh0,
    cu_seqlens,
    split_offsets,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STORE_MMA_STATE: tl.constexpr,
    FUSE_HDH_LAST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = i_n * NS

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # [BK, BV] accumulator; STATE_V_FIRST only flips the stored state's HBM layout to [V, K], applied at the load/store below.
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dht = tl.make_block_ptr(dht + i_nh * K*V, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            b_dh += tl.trans(tl.load(p_dht, boundary_check=(0, 1))).to(tl.float32)
        else:
            p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        i_s = i_t // (BS // BT)
        o_dh = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
        if STATE_V_FIRST:
            p_dh = tl.make_block_ptr(dh + o_dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            if STORE_MMA_STATE:
                p_dh_mma = tl.make_block_ptr(dh_mma + o_dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            if FUSE_HDH_LAST:
                p_h_for_hdh = tl.make_block_ptr(
                    h_for_hdh + o_dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
                )
        else:
            p_dh = tl.make_block_ptr(dh + o_dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            if STORE_MMA_STATE:
                p_dh_mma = tl.make_block_ptr(dh_mma + o_dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            if FUSE_HDH_LAST:
                p_h_for_hdh = tl.make_block_ptr(
                    h_for_hdh + o_dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
                )

        if i_t % (BS // BT) == 0:
            tl.store(p_dh, (tl.trans(b_dh) if STATE_V_FIRST else b_dh).to(p_dh.dtype.element_ty), boundary_check=(0, 1))
            if STORE_MMA_STATE:
                tl.store(
                    p_dh_mma,
                    (tl.trans(b_dh) if STATE_V_FIRST else b_dh).to(p_dh_mma.dtype.element_ty),
                    boundary_check=(0, 1),
                )
            if FUSE_HDH_LAST:
                b_h_for_hdh = tl.load(p_h_for_hdh, boundary_check=(0, 1)).to(tl.float32)
                if STATE_V_FIRST:
                    b_h_for_hdh = tl.trans(b_h_for_hdh)
                tl.atomic_add(
                    hdh_last + (boh + i_s) * H + i_h,
                    tl.sum(b_h_for_hdh * b_dh),
                    sem='relaxed',
                )
        last_idx = min(i_t * BT + BT, T) - 1
        # [BK, BT]
        p_q = tl.make_block_ptr(q + (bos*HQ + i_hq) * K, (K, T), (1, HQ*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + (bos*HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if USE_G:
            p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_q = (b_q * exp2(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp2(b_g_last)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_q = (b_q * exp2(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp2(b_g_last)

        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_q = (b_q * exp2(b_gk)).to(b_q.dtype)
            b_dh *= exp2(b_gk_last)[:, None]

        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_do = (b_do * exp2(b_gv))
            b_dh *= exp2(b_gv_last)[None, :]

        b_dh += tl.dot(b_q, b_do.to(b_q.dtype))

    if STORE_INITIAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            tl.store(p_dh0, tl.trans(b_dh).to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        else:
            p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _chunk_hdh_last_kernel(
    h,
    dh,
    hdh_last,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i_h = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for base in range(0, D, BLOCK):
        idx = base + offs
        mask = idx < D
        b_h = tl.load(h + i_h * D + idx, mask=mask, other=0.).to(tl.float32)
        b_dh = tl.load(dh + i_h * D + idx, mask=mask, other=0.).to(tl.float32)
        acc += b_h * b_dh

    tl.store(hdh_last + i_h, tl.sum(acc, axis=0))


def chunk_hdh_last(h: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    h_flat = h.reshape(-1, h.shape[-2] * h.shape[-1])
    dh_flat = dh.reshape(-1, dh.shape[-2] * dh.shape[-1])
    D = h_flat.shape[1]
    block = min(triton.next_power_of_2(D), 1024)
    hdh_last = torch.empty(h_flat.shape[0], dtype=torch.float32, device=h.device)
    _chunk_hdh_last_kernel[(h_flat.shape[0],)](
        h_flat,
        dh_flat,
        hdh_last,
        D,
        BLOCK=block,
        num_warps=8 if block >= 1024 else 4,
    )
    return hdh_last


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    output_mma_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()

    # `state_v_first` stores the states in V-first `[V, K]` layout instead of `[K, V]`
    state_shape = (V, K) if state_v_first else (K, V)
    h = k.new_empty(B, NS, H, *state_shape, dtype=k.dtype if not states_in_fp32 else torch.float)
    h_mma = k.new_empty(B, NS, H, *state_shape, dtype=k.dtype) if output_mma_state else None
    ht = k.new_empty(N, H, *state_shape, dtype=torch.float) if output_final_state else None
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_fwd_kernel_h[grid](
        k=k,
        v=v,
        h=h,
        h_mma=h_mma,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        STATE_V_FIRST=state_v_first,
    )
    if output_mma_state:
        return h, ht, h_mma
    return h, ht


def chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    output_mma_state: bool = False,
    h_for_hdh: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NG: number of groups in GQA
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()
    NG = HQ // H

    # `state_v_first` stores the states in V-first `[V, K]` layout instead of `[K, V]`
    state_shape = (V, K) if state_v_first else (K, V)
    dh = k.new_empty(B, NS, HQ, *state_shape, dtype=k.dtype if not states_in_fp32 else torch.float)
    dh_mma = k.new_empty(B, NS, HQ, *state_shape, dtype=q.dtype) if output_mma_state else None
    hdh_last = q.new_zeros(B, NS, H, dtype=torch.float32) if h_for_hdh is not None else None
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_bwd_kernel_dh[grid](
        q=q,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        do=do,
        dh=dh,
        dh_mma=dh_mma,
        h_for_hdh=h_for_hdh,
        hdh_last=hdh_last,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        scale=scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        NG=NG,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        STATE_V_FIRST=state_v_first,
    )
    if output_mma_state:
        if hdh_last is not None:
            return dh, dh0, dh_mma, hdh_last
        return dh, dh0, dh_mma
    return dh, dh0
