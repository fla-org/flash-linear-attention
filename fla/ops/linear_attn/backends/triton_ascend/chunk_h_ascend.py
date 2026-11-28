# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_fwd_h / chunk_bwd_dh for triton-ascend on Ascend NPU (GLA-style state).

Ascend requires host-specialized ``NT``, and the chunk loop is written as
``tl.range(NT)`` (a real loop, not ``tl.static_range``): ``static_range`` fully
unrolls, and the Ascend CV pipeline then allocates one tightly-coupled UB buffer
per unrolled iteration, so UB demand grows with ``NT`` until BiShengIR
``PlanMemory`` aborts (repro: B=1 T=2048 BT=64 -> NT=32). A real loop keeps a
single rotating buffer set while ``NT`` stays a compile-time constant.
A dynamic ``for i_t in range(tl.cdiv(T, BT))`` under-iterates when ``T`` is
unspecialized. The kernels below only handle equal-length inputs; varlen inputs
are split per sequence on the host (see ``chunk_bwd_dh_npu``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .ascend_limits import DH_UB_BUDGET_BYTES as _DH_UB_BUDGET_BYTES
from .ascend_ub_manager import launch_grid_chunked
from .utils import exp2, input_guard, prepare_chunk_offsets, zero_acc

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


# Per-program on-chip budget (bytes) that ``chunk_bwd_kernel_dh_npu`` is compiled
# against.  Exceeding it is *not* reported as a compile-time overflow for this
# kernel: BiShengIR emits a bulk MTE transfer that writes past the allocated
# region and every program traps at run time, which the runtime surfaces as
# "aicore error exception ... mte error info: <nonzero>" (error code 507015).
# The value lives in kernel/ascend_limits.py together with the other budgets.


def _chunk_bwd_dh_tile_size(
    K: int, V: int, BT: int, has_dht: bool, element_size: int
) -> tuple[int, int]:
    """Pick BK/BV for ``chunk_bwd_kernel_dh_npu`` on Ascend.

    The kernel keeps ``b_q[BK, BT]`` and ``b_do[BT, BV]`` (element dtype) plus
    the fp32 ``b_dh[BK, BV]`` accumulator live per program.  When a final-state
    gradient seeds the accumulator a fourth ``[BK, BV]`` tile is live as well: at
    BK=BV=BT=64 in fp32 that reaches the on-chip budget and every program traps
    (repro: B=2 T=256 H=4 K=V=64 fp32 with ``dht`` non-None; the same shape with
    ``dht=None`` runs fine).  Halve the K/V tile for that case only - it splits
    the K/V reduction, so the result is exact.

    The fp16/bfloat16 path stays at BK=BV=64 on purpose: its tiles are half the
    size, and shrinking them would push the kernel's ``tl.dot`` tiles onto the
    16-bit dot path that miscomputes when the reduction extent is smaller than
    the right operand width.
    """
    BK, BV = _chunk_h_tile_size(K, V)
    if not has_dht:
        return BK, BV
    while (
        BK > 16
        and BV > 16
        and ((BT * (BK + BV) + BK * BV) * element_size + BK * BV * 4)
        >= _DH_UB_BUDGET_BYTES
    ):
        BK //= 2
        BV //= 2
    return BK, BV


@triton.heuristics(
    {
        "STORE_INITIAL_STATE_GRADIENT": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "K_OFFSET", "V_OFFSET", "NH_OFFSET"])
def chunk_bwd_kernel_dh_npu(
    q,
    g,
    g_gamma,
    gk,
    gv,
    do,
    dh,
    dht,
    dh0,
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
    NT: tl.constexpr,
    NG: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    K_OFFSET,
    V_OFFSET,
    NH_OFFSET,
):
    i_k = tl.program_id(0) + K_OFFSET
    i_v = tl.program_id(1) + V_OFFSET
    i_nh = tl.program_id(2).to(tl.int64) + NH_OFFSET
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    bos = tl.cast(i_n, tl.int64) * T
    boh = tl.cast(i_n, tl.int64) * tl.cdiv(T, BS)

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    b_dh = zero_acc(BK, BV)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    if USE_FINAL_STATE_GRADIENT:
        dht_base = i_nh * K * V
        if STATE_V_FIRST:
            p_dht = dht + dht_base + o_v[:, None].to(tl.int64) * K + o_k[None, :]
            b_dh += tl.trans(
                tl.load(p_dht, mask=(o_v[:, None] < V) & (o_k[None, :] < K), other=0.0)
            ).to(tl.float32)
        else:
            p_dht = dht + dht_base + o_k[:, None].to(tl.int64) * V + o_v[None, :]
            b_dh += tl.load(
                p_dht, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0.0
            ).to(tl.float32)

    for step in tl.range(NT):
        i_t = NT - 1 - step
        i_s = i_t // (BS // BT)
        o_dh = ((boh + i_s) * H + i_h) * K * V
        if STATE_V_FIRST:
            p_dh = dh + o_dh + o_v[:, None].to(tl.int64) * K + o_k[None, :]
            m_dh = (o_v[:, None] < V) & (o_k[None, :] < K)
        else:
            p_dh = dh + o_dh + o_k[:, None].to(tl.int64) * V + o_v[None, :]
            m_dh = (o_k[:, None] < K) & (o_v[None, :] < V)

        if i_t % (BS // BT) == 0:
            tl.store(
                p_dh,
                (tl.trans(b_dh) if STATE_V_FIRST else b_dh).to(p_dh.dtype.element_ty),
                mask=m_dh,
            )

        last_idx = min(i_t * BT + BT, T) - 1
        o_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
        m_t = o_t < T
        qv_base = bos * HQ + i_hq
        p_q = q + qv_base * K + o_k[:, None] + o_t[None, :] * (HQ * K)
        p_do = do + qv_base * V + o_t[:, None] * (HQ * V) + o_v[None, :]
        b_q = (
            tl.load(p_q, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0).to(
                tl.float32
            )
            * scale
        )
        b_do = tl.load(p_do, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0).to(
            tl.float32
        )

        if USE_G:
            p_g = g + bos * H + o_t * H + i_h
            b_g_last = tl.load(g + bos * H + last_idx.to(tl.int64) * H + i_h).to(
                tl.float32
            )
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_q = b_q * exp2(b_g)[None, :]
            b_dh *= exp2(b_g_last)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_q = b_q * exp2(b_g)[None, :]
            b_dh *= exp2(b_g_last)

        if USE_GK:
            kv_base = (bos * H + i_h) * K
            p_gk = gk + kv_base + o_k[:, None] + o_t[None, :] * (H * K)
            p_gk_last = (
                gk
                + (bos + last_idx.to(tl.int64)) * (H * K)
                + i_h * K
                + i_k * BK
                + tl.arange(0, BK)
            )
            b_gk = tl.load(p_gk, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0).to(
                tl.float32
            )
            b_gk_last = tl.load(
                p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.0
            ).to(tl.float32)
            b_q = b_q * exp2(b_gk)
            b_dh *= exp2(b_gk_last)[:, None]

        if USE_GV:
            p_gv = gv + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
            p_gv_last = (
                gv
                + (bos + last_idx.to(tl.int64)) * (H * V)
                + i_h * V
                + i_v * BV
                + tl.arange(0, BV)
            )
            b_gv = tl.load(p_gv, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0).to(
                tl.float32
            )
            b_gv_last = tl.load(
                p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.0
            ).to(tl.float32)
            b_do = b_do * exp2(b_gv)
            b_dh *= exp2(b_gv_last)[None, :]

        b_dh = tl.dot(b_q, b_do, b_dh)

    if STORE_INITIAL_STATE_GRADIENT:
        dh0_base = i_nh * K * V
        if STATE_V_FIRST:
            p_dh0 = dh0 + dh0_base + o_v[:, None].to(tl.int64) * K + o_k[None, :]
            tl.store(
                p_dh0,
                tl.trans(b_dh).to(p_dh0.dtype.element_ty),
                mask=(o_v[:, None] < V) & (o_k[None, :] < K),
            )
        else:
            p_dh0 = dh0 + dh0_base + o_k[:, None].to(tl.int64) * V + o_v[None, :]
            tl.store(
                p_dh0,
                b_dh.to(p_dh0.dtype.element_ty),
                mask=(o_k[:, None] < K) & (o_v[None, :] < V),
            )


def _slice_seq(x: torch.Tensor | None, bos: int, eos: int) -> torch.Tensor | None:
    return None if x is None else x[:, bos:eos]


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
    assert (
        BS % BT == 0
    ), f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"

    # packed varlen: run the equal-length kernel per sequence for complete state stores
    if cu_seqlens is not None:
        assert B == 1, "NPU varlen chunk_bwd_dh expects packed batch B=1"
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N = len(cu_seqlens) - 1
        NS = int(split_offsets[-1].item())
        state_shape = (V, K) if state_v_first else (K, V)
        dh = k.new_zeros(
            B, NS, HQ, *state_shape, dtype=torch.float if states_in_fp32 else k.dtype
        )
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
            i_n_end = i_n + 1
            boh_end = boh + n_i
            dh_i, dh0_i = chunk_bwd_dh_npu(
                q=_slice_seq(q, bos, eos),
                k=_slice_seq(k, bos, eos),
                v=_slice_seq(v, bos, eos),
                do=_slice_seq(do, bos, eos),
                h0=None if h0 is None else h0[i_n:i_n_end],
                dht=None if dht is None else dht[i_n:i_n_end],
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
            dh[:, boh:boh_end] = dh_i
            if dh0 is not None and dh0_i is not None:
                dh0[i_n].copy_(dh0_i[0])
        return dh, dh0

    N, NS = B, triton.cdiv(T, BS)
    NG = HQ // H
    NT = triton.cdiv(T, BT)

    # Pad the time axis up to a whole number of chunks so the tail chunk has no
    # masked row.  On Ascend a masked-off lane still forms its address and the
    # out-of-range data leaks into the fp32 accumulator, so `dh` comes back NaN
    # whenever T is not a multiple of BT (repro: T=100/BT=64 gives NaN in `dh`,
    # while T=64 and T=128 are clean).  Clamping the lane index instead (the
    # trick used in chunk_h.py) traps the device for this kernel, so pad here.
    # The padded tokens are zeros, so they contribute nothing to the state
    # recurrence nor to the gradient.  `NS` above stays correct because T_pad and
    # T always fall in the same `BS`-sized bucket.
    T_pad = NT * BT
    if T_pad != T:
        pad = T_pad - T

        def _pad_time(x):
            if x is None:
                return None
            return torch.cat([x, x.new_zeros(B, pad, *x.shape[2:])], dim=1)

        q = _pad_time(q)
        do = _pad_time(do)
        g = _pad_time(g)
        gk = _pad_time(gk)
        gv = _pad_time(gv)
        T = T_pad

    state_shape = (V, K) if state_v_first else (K, V)
    dh = k.new_zeros(
        B, NS, HQ, *state_shape, dtype=torch.float if states_in_fp32 else k.dtype
    )
    dh0 = torch.zeros_like(h0, dtype=torch.float) if h0 is not None else None

    BK, BV = _chunk_bwd_dh_tile_size(K, V, BT, dht is not None, q.element_size())
    launch_grid_chunked(
        chunk_bwd_kernel_dh_npu,
        (triton.cdiv(K, BK), triton.cdiv(V, BV), N * HQ),
        offset_keys=("K_OFFSET", "V_OFFSET", "NH_OFFSET"),
        kernel_kwargs=dict(
            q=q,
            g=g,
            g_gamma=g_gamma,
            gk=gk,
            gv=gv,
            do=do,
            dh=dh,
            dht=dht,
            dh0=dh0,
            scale=scale,
            T=T,
            HQ=HQ,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            NT=NT,
            NG=NG,
            USE_G=g is not None,
            USE_G_GAMMA=g_gamma is not None,
            USE_GK=gk is not None,
            USE_GV=gv is not None,
            STATE_V_FIRST=state_v_first,
            K_OFFSET=0,
            V_OFFSET=0,
            NH_OFFSET=0,
        ),
    )
    return dh, dh0
