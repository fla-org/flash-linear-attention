# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_fwd_h / chunk_bwd_dh for triton-ascend on Ascend NPU (GLA-style state).

Ascend requires host-specialized ``NT`` + ``tl.static_range(NT)``. Dynamic
``for i_t in range(tl.cdiv(T, BT))`` under-iterates when ``T`` is unspecialized.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM, max_grid_axis_chunks

_NUM_WARPS = 4
_LAUNCH_BLOCK_BUDGET = 4096
# Fixed tiles: matches validated probe; avoids UB/autotune variance on Ascend.
_BK = 64
_BV = 64


def _max_nt(T: int, BT: int, cu_seqlens: torch.Tensor | None) -> int:
    if cu_seqlens is None:
        return triton.cdiv(T, BT)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return int(triton.cdiv(int(lengths.max().item()), BT))


def _launch_kv_nh(kernel, *, nk: int, nv: int, nh: int, kernel_kwargs: dict) -> None:
    budget = _LAUNCH_BLOCK_BUDGET
    nk_step = nk if nk * nv * nh <= budget else max(1, budget // max(nv * nh, 1))
    for k_off in range(0, nk, nk_step):
        k_len = min(nk_step, nk - k_off)
        kernel_kwargs['K_OFFSET'] = k_off
        nv_budget = max(1, budget // max(k_len * nh, 1))
        nv_step = min(nv_budget, max_grid_axis_chunks(nv, k_len * nh, max_grid=ASCEND_MAX_GRID_DIM))
        for v_off in range(0, nv, nv_step):
            v_len = min(nv_step, nv - v_off)
            kernel_kwargs['V_OFFSET'] = v_off
            nh_budget = max(1, budget // max(k_len * v_len, 1))
            nh_step = min(nh_budget, max_grid_axis_chunks(nh, k_len * v_len, max_grid=ASCEND_MAX_GRID_DIM))
            for nh_off in range(0, nh, nh_step):
                nh_len = min(nh_step, nh - nh_off)
                kernel_kwargs['NH_OFFSET'] = nh_off
                kernel[(k_len, v_len, nh_len)](num_warps=_NUM_WARPS, **kernel_kwargs)
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.synchronize()


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'K_OFFSET', 'V_OFFSET', 'NH_OFFSET'])
def chunk_fwd_kernel_h_npu(
    k, v, h, g, g_gamma, gk, gv, h0, ht,
    cu_seqlens, split_offsets, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr,
    USE_G: tl.constexpr, USE_G_GAMMA: tl.constexpr, USE_GK: tl.constexpr, USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr, STATE_V_FIRST: tl.constexpr,
    K_OFFSET, V_OFFSET, NH_OFFSET,
):
    i_k = tl.program_id(0) + K_OFFSET
    i_v = tl.program_id(1) + V_OFFSET
    i_nh = tl.program_id(2).to(tl.int64) + NH_OFFSET
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT_SEQ = tl.cdiv(T, BT)
        boh = tl.load(split_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT_SEQ = NT
        boh = i_n * tl.cdiv(T, BS)
    NTS = BS // BT

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0 = h0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
            b_h = tl.trans(tl.load(p_h0, mask=(o_v[:, None] < V) & (o_k[None, :] < K), other=0.0)).to(tl.float32)
        else:
            p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
            b_h = tl.load(p_h0, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0.0).to(tl.float32)

    for i_t in tl.static_range(NT):
        # Always execute body; inactive (varlen pad) iterations contribute zeros via masks.
        active = i_t < NT_SEQ
        i_s = i_t // NTS
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = (o_t < T) & active
        p_k = k + (bos * H + i_h) * K + o_k[:, None] + o_t[None, :] * (H * K)
        p_v = v + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K * V
        if STATE_V_FIRST:
            p_h = h + o_h + o_v[:, None] * K + o_k[None, :]
            m_h = (o_v[:, None] < V) & (o_k[None, :] < K) & active
        else:
            p_h = h + o_h + o_k[:, None] * V + o_v[None, :]
            m_h = (o_k[:, None] < K) & (o_v[None, :] < V) & active

        if i_t % NTS == 0:
            tl.store(p_h, (tl.trans(b_h) if STATE_V_FIRST else b_h).to(p_h.dtype.element_ty), mask=m_h)

        # Force fp32 recurrence on Ascend (bf16 tl.dot is less stable than CUDA).
        b_k = tl.load(p_k, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0).to(tl.float32)
        last_idx = min((i_t + 1) * BT, T) - 1
        last_idx = max(last_idx, 0)

        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
            p_g = g + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T) & active, other=0.).to(tl.float32)
            b_h *= tl.where(active, exp2(b_g_last), 1.0)
            b_v = b_v * exp2(b_g_last - b_g)[:, None]

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, max(T - i_t * BT, 0))
            b_h *= tl.where(active, exp2(b_g_last), 1.0)
            b_v = b_v * exp2(b_g_last - b_g)[:, None]

        if USE_GK:
            p_gk = gk + (bos * H + i_h) * K + o_k[:, None] + o_t[None, :] * (H * K)
            p_gk_last = gk + (bos + last_idx) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.).to(tl.float32)
            b_gk = tl.load(p_gk, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0).to(tl.float32)
            b_h *= tl.where(active, exp2(b_gk_last), 1.0)[:, None]
            b_k = b_k * exp2(b_gk_last[:, None] - b_gk)

        if USE_GV:
            p_gv = gv + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
            p_gv_last = gv + (bos + last_idx) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0).to(tl.float32)
            b_h *= tl.where(active, exp2(b_gv_last), 1.0)[None, :]
            b_v = b_v * exp2(b_gv_last[None, :] - b_gv)

        b_h += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, tl.trans(b_h).to(p_ht.dtype.element_ty), mask=(o_v[:, None] < V) & (o_k[None, :] < K))
        else:
            p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=(o_k[:, None] < K) & (o_v[None, :] < V))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'K_OFFSET', 'V_OFFSET', 'NH_OFFSET'])
def chunk_bwd_kernel_dh_npu(
    q, g, g_gamma, gk, gv, do, dh, dht, dh0,
    cu_seqlens, split_offsets, scale, T,
    HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr, NG: tl.constexpr,
    USE_G: tl.constexpr, USE_G_GAMMA: tl.constexpr, USE_GK: tl.constexpr, USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr, USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr, STATE_V_FIRST: tl.constexpr,
    K_OFFSET, V_OFFSET, NH_OFFSET,
):
    i_k = tl.program_id(0) + K_OFFSET
    i_v = tl.program_id(1) + V_OFFSET
    i_nh = tl.program_id(2).to(tl.int64) + NH_OFFSET
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT_SEQ = tl.cdiv(T, BT)
        boh = tl.load(split_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT_SEQ = NT
        boh = i_n * tl.cdiv(T, BS)

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dht = dht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
            b_dh += tl.trans(tl.load(p_dht, mask=(o_v[:, None] < V) & (o_k[None, :] < K), other=0.0)).to(tl.float32)
        else:
            p_dht = dht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
            b_dh += tl.load(p_dht, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0.0).to(tl.float32)

    for step in tl.static_range(NT):
        i_t = NT - 1 - step
        active = i_t < NT_SEQ
        i_s = i_t // (BS // BT)
        o_dh = ((boh + i_s) * H + i_h).to(tl.int64) * K * V
        if STATE_V_FIRST:
            p_dh = dh + o_dh + o_v[:, None] * K + o_k[None, :]
            m_dh = (o_v[:, None] < V) & (o_k[None, :] < K) & active
        else:
            p_dh = dh + o_dh + o_k[:, None] * V + o_v[None, :]
            m_dh = (o_k[:, None] < K) & (o_v[None, :] < V) & active

        if i_t % (BS // BT) == 0:
            tl.store(p_dh, (tl.trans(b_dh) if STATE_V_FIRST else b_dh).to(p_dh.dtype.element_ty), mask=m_dh)

        last_idx = min(i_t * BT + BT, T) - 1
        last_idx = max(last_idx, 0)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = (o_t < T) & active
        p_q = q + (bos * HQ + i_hq) * K + o_k[:, None] + o_t[None, :] * (HQ * K)
        p_do = do + (bos * HQ + i_hq) * V + o_t[:, None] * (HQ * V) + o_v[None, :]
        b_q = tl.load(p_q, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0).to(tl.float32) * scale
        b_do = tl.load(p_do, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0).to(tl.float32)

        if USE_G:
            p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h).to(tl.float32)
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T) & active, other=0.).to(tl.float32)
            b_q = b_q * exp2(b_g)[None, :]
            b_dh *= tl.where(active, exp2(b_g_last), 1.0)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, max(T - i_t * BT, 0))
            b_q = b_q * exp2(b_g)[None, :]
            b_dh *= tl.where(active, exp2(b_g_last), 1.0)

        if USE_GK:
            p_gk = gk + (bos * H + i_h) * K + o_k[:, None] + o_t[None, :] * (H * K)
            p_gk_last = gk + (bos + last_idx) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
            b_gk = tl.load(p_gk, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0).to(tl.float32)
            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.).to(tl.float32)
            b_q = b_q * exp2(b_gk)
            b_dh *= tl.where(active, exp2(b_gk_last), 1.0)[:, None]

        if USE_GV:
            p_gv = gv + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
            p_gv_last = gv + (bos + last_idx) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
            b_gv = tl.load(p_gv, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0).to(tl.float32)
            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.).to(tl.float32)
            b_do = b_do * exp2(b_gv)
            b_dh *= tl.where(active, exp2(b_gv_last), 1.0)[None, :]

        b_dh += tl.dot(b_q, b_do)

    if STORE_INITIAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dh0 = dh0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
            tl.store(p_dh0, tl.trans(b_dh).to(p_dh0.dtype.element_ty), mask=(o_v[:, None] < V) & (o_k[None, :] < K))
        else:
            p_dh0 = dh0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
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
    assert K <= 256 and V <= 256, "NPU chunk_h currently supports K,V <= 256"

    # Ascend fused IS_VARLEN path can leave partial NaNs in `ht` while `h` stays
    # correct. Route each packed sequence through the validated equal-length path.
    if cu_seqlens is not None:
        assert B == 1, "NPU varlen chunk_h expects packed batch B=1"
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N = len(cu_seqlens) - 1
        NS = int(split_offsets[-1].item())
        state_shape = (V, K) if state_v_first else (K, V)
        # zeros: tests/conftest poisons empty*() with NaN; partial stores must not leak.
        h = k.new_zeros(B, NS, H, *state_shape, dtype=torch.float)
        ht = k.new_zeros(N, H, *state_shape, dtype=torch.float) if output_final_state else None
        for i_n in range(N):
            bos = int(cu_seqlens[i_n].item())
            eos = int(cu_seqlens[i_n + 1].item())
            boh = int(split_offsets[i_n].item())
            n_i = int(split_offsets[i_n + 1].item()) - boh
            if eos <= bos or n_i <= 0:
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

    N, NS, split_offsets = B, triton.cdiv(T, BS), None
    NT = _max_nt(T, BT, None)
    state_shape = (V, K) if state_v_first else (K, V)
    # zeros: tests/conftest poisons empty*() with NaN; partial stores must not leak.
    h = k.new_zeros(B, NS, H, *state_shape, dtype=torch.float)
    ht = k.new_zeros(N, H, *state_shape, dtype=torch.float) if output_final_state else None

    BK, BV = _BK, _BV
    nk, nv, nh = triton.cdiv(K, BK), triton.cdiv(V, BV), N * H
    _launch_kv_nh(
        chunk_fwd_kernel_h_npu,
        nk=nk, nv=nv, nh=nh,
        kernel_kwargs=dict(
            k=k, v=v, h=h, g=g, g_gamma=g_gamma, gk=gk, gv=gv, h0=h0, ht=ht,
            cu_seqlens=None, split_offsets=None, T=T,
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
    assert K <= 256 and V <= 256, "NPU chunk_h currently supports K,V <= 256"

    if cu_seqlens is not None:
        assert B == 1, "NPU varlen chunk_bwd_dh expects packed batch B=1"
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N = len(cu_seqlens) - 1
        NS = int(split_offsets[-1].item())
        state_shape = (V, K) if state_v_first else (K, V)
        dh = k.new_zeros(B, NS, HQ, *state_shape, dtype=torch.float)
        dh0 = torch.zeros_like(h0, dtype=torch.float) if h0 is not None else None
        for i_n in range(N):
            bos = int(cu_seqlens[i_n].item())
            eos = int(cu_seqlens[i_n + 1].item())
            boh = int(split_offsets[i_n].item())
            n_i = int(split_offsets[i_n + 1].item()) - boh
            if eos <= bos or n_i <= 0:
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

    N, NS, split_offsets = B, triton.cdiv(T, BS), None
    NG = HQ // H
    NT = _max_nt(T, BT, None)

    state_shape = (V, K) if state_v_first else (K, V)
    dh = k.new_zeros(B, NS, HQ, *state_shape, dtype=torch.float)
    dh0 = torch.zeros_like(h0, dtype=torch.float) if h0 is not None else None

    BK, BV = _BK, _BV
    nk, nv, nh = triton.cdiv(K, BK), triton.cdiv(V, BV), N * HQ
    _launch_kv_nh(
        chunk_bwd_kernel_dh_npu,
        nk=nk, nv=nv, nh=nh,
        kernel_kwargs=dict(
            q=q, g=g, g_gamma=g_gamma, gk=gk, gv=gv, do=do, dh=dh, dht=dht, dh0=dh0,
            cu_seqlens=None, split_offsets=None, scale=scale, T=T,
            HQ=HQ, H=H, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV, NT=NT, NG=NG,
            USE_G=g is not None, USE_G_GAMMA=g_gamma is not None,
            USE_GK=gk is not None, USE_GV=gv is not None,
            STATE_V_FIRST=state_v_first,
            K_OFFSET=0, V_OFFSET=0, NH_OFFSET=0,
        ),
    )
    return dh, dh0
