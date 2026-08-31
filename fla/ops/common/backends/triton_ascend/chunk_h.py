# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_fwd_h / chunk_bwd_dh for triton-ascend on Ascend NPU (GLA-style state).

Ascend requires host-specialized ``NT`` + ``tl.static_range(NT)``. Dynamic
``for i_t in range(tl.cdiv(T, BT))`` under-iterates when ``T`` is unspecialized.
The kernels below only handle equal-length inputs; varlen inputs are split per
sequence on the host (see ``chunk_fwd_h_npu``/``chunk_bwd_dh_npu``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import launch_grid_chunked

# Fixed tiles: avoids autotune picking UB-overflowing configs on Ascend.
_BK = 64
_BV = 64


def _chunk_h_tile_size(K: int, V: int) -> tuple[int, int]:
    """Pick BK/BV for chunk_h on Ascend.

    BK=BV=64 overflows UB when K,V>=256 and USE_GK loads extra fp32 tiles
    (b_k/b_gk/b_h/b_v), which surfaces as MTE DDR OOB for large B*H.
    """
    if K > 128 or V > 128:
        return 32, 32
    return _BK, _BV


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
})
@triton.jit(do_not_specialize=['T', 'K_OFFSET', 'V_OFFSET', 'NH_OFFSET'])
def chunk_fwd_kernel_h_npu(
    k, v, h, g, g_gamma, gk, gv, h0, ht, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr,
    USE_G: tl.constexpr, USE_G_GAMMA: tl.constexpr, USE_GK: tl.constexpr, USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    K_OFFSET, V_OFFSET, NH_OFFSET,
):
    i_k = tl.program_id(0) + K_OFFSET
    i_v = tl.program_id(1) + V_OFFSET
    i_nh = tl.program_id(2).to(tl.int64) + NH_OFFSET
    i_n, i_h = i_nh // H, i_nh % H
    bos = (i_n * T).to(tl.int64)
    boh = (i_n * tl.cdiv(T, BS)).to(tl.int64)
    NTS = BS // BT

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    if USE_INITIAL_STATE:
        h0_base = (i_nh * K * V).to(tl.int64)
        if STATE_V_FIRST:
            p_h0 = h0 + h0_base + o_v[:, None].to(tl.int64) * K + o_k[None, :]
            b_h = tl.trans(tl.load(p_h0, mask=(o_v[:, None] < V) & (o_k[None, :] < K), other=0.0)).to(tl.float32)
        else:
            p_h0 = h0 + h0_base + o_k[:, None].to(tl.int64) * V + o_v[None, :]
            b_h = tl.load(p_h0, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0.0).to(tl.float32)

    for i_t in tl.static_range(NT):
        i_s = i_t // NTS
        o_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
        m_t = o_t < T
        kv_base = (bos * H + i_h).to(tl.int64) * K
        p_k = k + kv_base + o_k[:, None] + o_t[None, :] * (H * K)
        p_v = v + (bos * H + i_h).to(tl.int64) * V + o_t[:, None] * (H * V) + o_v[None, :]

        o_h = ((boh + i_s.to(tl.int64)) * H + i_h).to(tl.int64) * K * V
        if STATE_V_FIRST:
            p_h = h + o_h + o_v[:, None].to(tl.int64) * K + o_k[None, :]
            m_h = (o_v[:, None] < V) & (o_k[None, :] < K)
        else:
            p_h = h + o_h + o_k[:, None].to(tl.int64) * V + o_v[None, :]
            m_h = (o_k[:, None] < K) & (o_v[None, :] < V)

        if i_t % NTS == 0:
            tl.store(p_h, (tl.trans(b_h) if STATE_V_FIRST else b_h).to(p_h.dtype.element_ty), mask=m_h)

        # Force fp32 recurrence on Ascend (bf16 tl.dot is less stable than CUDA).
        b_k = tl.load(p_k, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0).to(tl.float32)
        last_idx = min((i_t + 1) * BT, T) - 1

        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx.to(tl.int64) * H + i_h).to(tl.float32)
            p_g = g + bos * H + o_t * H + i_h
            b_g = tl.load(p_g, mask=m_t, other=0.).to(tl.float32)
            b_h *= exp2(b_g_last)
            b_v = b_v * exp2(b_g_last - b_g)[:, None]

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_h *= exp2(b_g_last)
            b_v = b_v * exp2(b_g_last - b_g)[:, None]

        if USE_GK:
            p_gk = gk + kv_base + o_k[:, None] + o_t[None, :] * (H * K)
            p_gk_last = gk + (bos + last_idx.to(tl.int64)) * (H * K) + i_h * K + i_k * BK + tl.arange(0, BK)
            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.).to(tl.float32)
            b_gk = tl.load(p_gk, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0).to(tl.float32)
            b_h *= exp2(b_gk_last)[:, None]
            b_k = b_k * exp2(b_gk_last[:, None] - b_gk)

        if USE_GV:
            p_gv = gv + (bos * H + i_h).to(tl.int64) * V + o_t[:, None] * (H * V) + o_v[None, :]
            p_gv_last = gv + (bos + last_idx.to(tl.int64)) * (H * V) + i_h * V + i_v * BV + tl.arange(0, BV)
            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0).to(tl.float32)
            b_h *= exp2(b_gv_last)[None, :]
            b_v = b_v * exp2(b_gv_last[None, :] - b_gv)

        b_h += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        ht_base = (i_nh * K * V).to(tl.int64)
        if STATE_V_FIRST:
            p_ht = ht + ht_base + o_v[:, None].to(tl.int64) * K + o_k[None, :]
            tl.store(p_ht, tl.trans(b_h).to(p_ht.dtype.element_ty), mask=(o_v[:, None] < V) & (o_k[None, :] < K))
        else:
            p_ht = ht + ht_base + o_k[:, None].to(tl.int64) * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=(o_k[:, None] < K) & (o_v[None, :] < V))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
})
@triton.jit(do_not_specialize=['T', 'K_OFFSET', 'V_OFFSET', 'NH_OFFSET'])
def chunk_bwd_kernel_dh_npu(
    q, g, g_gamma, gk, gv, do, dh, dht, dh0,
    scale, T,
    HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr, NG: tl.constexpr,
    USE_G: tl.constexpr, USE_G_GAMMA: tl.constexpr, USE_GK: tl.constexpr, USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr, USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    K_OFFSET, V_OFFSET, NH_OFFSET,
):
    i_k = tl.program_id(0) + K_OFFSET
    i_v = tl.program_id(1) + V_OFFSET
    i_nh = tl.program_id(2).to(tl.int64) + NH_OFFSET
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    bos = (i_n * T).to(tl.int64)
    boh = (i_n * tl.cdiv(T, BS)).to(tl.int64)

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    if USE_FINAL_STATE_GRADIENT:
        dht_base = (i_nh * K * V).to(tl.int64)
        if STATE_V_FIRST:
            p_dht = dht + dht_base + o_v[:, None].to(tl.int64) * K + o_k[None, :]
            b_dh += tl.trans(tl.load(p_dht, mask=(o_v[:, None] < V) & (o_k[None, :] < K), other=0.0)).to(tl.float32)
        else:
            p_dht = dht + dht_base + o_k[:, None].to(tl.int64) * V + o_v[None, :]
            b_dh += tl.load(p_dht, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0.0).to(tl.float32)

    for step in tl.static_range(NT):
        i_t = NT - 1 - step
        i_s = i_t // (BS // BT)
        o_dh = ((boh + i_s.to(tl.int64)) * H + i_h).to(tl.int64) * K * V
        if STATE_V_FIRST:
            p_dh = dh + o_dh + o_v[:, None].to(tl.int64) * K + o_k[None, :]
            m_dh = (o_v[:, None] < V) & (o_k[None, :] < K)
        else:
            p_dh = dh + o_dh + o_k[:, None].to(tl.int64) * V + o_v[None, :]
            m_dh = (o_k[:, None] < K) & (o_v[None, :] < V)

        if i_t % (BS // BT) == 0:
            tl.store(p_dh, (tl.trans(b_dh) if STATE_V_FIRST else b_dh).to(p_dh.dtype.element_ty), mask=m_dh)

        last_idx = min(i_t * BT + BT, T) - 1
        o_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
        m_t = o_t < T
        qv_base = (bos * HQ + i_hq).to(tl.int64)
        p_q = q + qv_base * K + o_k[:, None] + o_t[None, :] * (HQ * K)
        p_do = do + qv_base * V + o_t[:, None] * (HQ * V) + o_v[None, :]
        b_q = tl.load(p_q, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0).to(tl.float32) * scale
        b_do = tl.load(p_do, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0).to(tl.float32)

        if USE_G:
            p_g = g + bos * H + o_t * H + i_h
            b_g_last = tl.load(g + bos * H + last_idx.to(tl.int64) * H + i_h).to(tl.float32)
            b_g = tl.load(p_g, mask=m_t, other=0.).to(tl.float32)
            b_q = b_q * exp2(b_g)[None, :]
            b_dh *= exp2(b_g_last)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_q = b_q * exp2(b_g)[None, :]
            b_dh *= exp2(b_g_last)

        if USE_GK:
            kv_base = (bos * H + i_h).to(tl.int64) * K
            p_gk = gk + kv_base + o_k[:, None] + o_t[None, :] * (H * K)
            p_gk_last = gk + (bos + last_idx.to(tl.int64)) * (H * K) + i_h * K + i_k * BK + tl.arange(0, BK)
            b_gk = tl.load(p_gk, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0).to(tl.float32)
            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.).to(tl.float32)
            b_q = b_q * exp2(b_gk)
            b_dh *= exp2(b_gk_last)[:, None]

        if USE_GV:
            p_gv = gv + (bos * H + i_h).to(tl.int64) * V + o_t[:, None] * (H * V) + o_v[None, :]
            p_gv_last = gv + (bos + last_idx.to(tl.int64)) * (H * V) + i_h * V + i_v * BV + tl.arange(0, BV)
            b_gv = tl.load(p_gv, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0).to(tl.float32)
            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.).to(tl.float32)
            b_do = b_do * exp2(b_gv)
            b_dh *= exp2(b_gv_last)[None, :]

        b_dh += tl.dot(b_q, b_do)

    if STORE_INITIAL_STATE_GRADIENT:
        dh0_base = (i_nh * K * V).to(tl.int64)
        if STATE_V_FIRST:
            p_dh0 = dh0 + dh0_base + o_v[:, None].to(tl.int64) * K + o_k[None, :]
            tl.store(p_dh0, tl.trans(b_dh).to(p_dh0.dtype.element_ty), mask=(o_v[:, None] < V) & (o_k[None, :] < K))
        else:
            p_dh0 = dh0 + dh0_base + o_k[:, None].to(tl.int64) * V + o_v[None, :]
            tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=(o_k[:, None] < K) & (o_v[None, :] < V))


def _slice_seq(x: torch.Tensor | None, bos: int, eos: int) -> torch.Tensor | None:
    return None if x is None else x[:, bos:eos]


@input_guard
def chunk_fwd_h_npu(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"

    # packed varlen: run the equal-length kernel per sequence for complete state stores
    if cu_seqlens is not None:
        assert B == 1, "NPU varlen chunk_h expects packed batch B=1"
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N = len(cu_seqlens) - 1
        NS = int(split_offsets[-1].item())
        state_shape = (V, K) if state_v_first else (K, V)
        # zero-init: kernels may only partially store each tile
        h = k.new_zeros(B, NS, H, *state_shape, dtype=torch.float if states_in_fp32 else k.dtype)
        ht = k.new_zeros(N, H, *state_shape, dtype=torch.float) if output_final_state else None
        for i_n in range(N):
            bos = int(cu_seqlens[i_n].item())
            eos = int(cu_seqlens[i_n + 1].item())
            boh = int(split_offsets[i_n].item())
            n_i = int(split_offsets[i_n + 1].item()) - boh
            if eos <= bos or n_i <= 0:
                if ht is not None and h0 is not None:
                    ht[i_n].copy_(h0[i_n])
                continue
            h_i, ht_i = chunk_fwd_h_npu(
                k=_slice_seq(k, bos, eos),
                v=_slice_seq(v, bos, eos),
                g=_slice_seq(g, bos, eos),
                g_gamma=g_gamma,
                gk=_slice_seq(gk, bos, eos),
                gv=_slice_seq(gv, bos, eos),
                h0=None if h0 is None else h0[i_n:i_n + 1],
                output_final_state=output_final_state,
                state_v_first=state_v_first,
                cu_seqlens=None,
                chunk_size=chunk_size,
                split_size=split_size,
                states_in_fp32=states_in_fp32,
            )
            h[:, boh:boh + n_i] = h_i
            if ht is not None:
                ht[i_n].copy_(ht_i[0])
        return h, ht

    N, NS = B, triton.cdiv(T, BS)
    NT = triton.cdiv(T, BT)
    state_shape = (V, K) if state_v_first else (K, V)
    # zero-init: kernels may only partially store each tile
    h = k.new_zeros(B, NS, H, *state_shape, dtype=torch.float if states_in_fp32 else k.dtype)
    ht = k.new_zeros(N, H, *state_shape, dtype=torch.float) if output_final_state else None

    BK, BV = _chunk_h_tile_size(K, V)
    launch_grid_chunked(
        chunk_fwd_kernel_h_npu,
        (triton.cdiv(K, BK), triton.cdiv(V, BV), N * H),
        offset_keys=('K_OFFSET', 'V_OFFSET', 'NH_OFFSET'),
        kernel_kwargs=dict(
            k=k, v=v, h=h, g=g, g_gamma=g_gamma, gk=gk, gv=gv, h0=h0, ht=ht, T=T,
            H=H, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV, NT=NT,
            USE_G=g is not None, USE_G_GAMMA=g_gamma is not None,
            USE_GK=gk is not None, USE_GV=gv is not None,
            STATE_V_FIRST=state_v_first,
            K_OFFSET=0, V_OFFSET=0, NH_OFFSET=0,
        ),
    )
    return h, ht


@input_guard
def chunk_bwd_dh_npu(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"

    # packed varlen: run the equal-length kernel per sequence for complete state stores
    if cu_seqlens is not None:
        assert B == 1, "NPU varlen chunk_bwd_dh expects packed batch B=1"
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N = len(cu_seqlens) - 1
        NS = int(split_offsets[-1].item())
        state_shape = (V, K) if state_v_first else (K, V)
        dh = k.new_zeros(B, NS, HQ, *state_shape, dtype=torch.float if states_in_fp32 else k.dtype)
        dh0 = torch.zeros_like(h0, dtype=torch.float) if h0 is not None else None
        for i_n in range(N):
            bos = int(cu_seqlens[i_n].item())
            eos = int(cu_seqlens[i_n + 1].item())
            boh = int(split_offsets[i_n].item())
            n_i = int(split_offsets[i_n + 1].item()) - boh
            if eos <= bos or n_i <= 0:
                if dh0 is not None and dht is not None:
                    dh0[i_n].copy_(dht[i_n])
                continue
            dh_i, dh0_i = chunk_bwd_dh_npu(
                q=_slice_seq(q, bos, eos),
                k=_slice_seq(k, bos, eos),
                v=_slice_seq(v, bos, eos),
                do=_slice_seq(do, bos, eos),
                h0=None if h0 is None else h0[i_n:i_n + 1],
                dht=None if dht is None else dht[i_n:i_n + 1],
                scale=scale,
                g=_slice_seq(g, bos, eos),
                g_gamma=g_gamma,
                gk=_slice_seq(gk, bos, eos),
                gv=_slice_seq(gv, bos, eos),
                state_v_first=state_v_first,
                cu_seqlens=None,
                chunk_size=chunk_size,
                split_size=split_size,
                states_in_fp32=states_in_fp32,
            )
            dh[:, boh:boh + n_i] = dh_i
            if dh0 is not None and dh0_i is not None:
                dh0[i_n].copy_(dh0_i[0])
        return dh, dh0

    N, NS = B, triton.cdiv(T, BS)
    NG = HQ // H
    NT = triton.cdiv(T, BT)

    state_shape = (V, K) if state_v_first else (K, V)
    dh = k.new_zeros(B, NS, HQ, *state_shape, dtype=torch.float if states_in_fp32 else k.dtype)
    dh0 = torch.zeros_like(h0, dtype=torch.float) if h0 is not None else None

    BK, BV = _chunk_h_tile_size(K, V)
    launch_grid_chunked(
        chunk_bwd_kernel_dh_npu,
        (triton.cdiv(K, BK), triton.cdiv(V, BV), N * HQ),
        offset_keys=('K_OFFSET', 'V_OFFSET', 'NH_OFFSET'),
        kernel_kwargs=dict(
            q=q, g=g, g_gamma=g_gamma, gk=gk, gv=gv, do=do, dh=dh, dht=dht, dh0=dh0,
            scale=scale, T=T,
            HQ=HQ, H=H, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV, NT=NT, NG=NG,
            USE_G=g is not None, USE_G_GAMMA=g_gamma is not None,
            USE_GK=gk is not None, USE_GV=gv is not None,
            STATE_V_FIRST=state_v_first,
            K_OFFSET=0, V_OFFSET=0, NH_OFFSET=0,
        ),
    )
    return dh, dh0
