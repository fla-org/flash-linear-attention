# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.parallax.parallel import _block_size
from fla.ops.utils.op import exp2


@triton.jit(do_not_specialize=['Sq', 'Skv'])
def parallax_decode_kernel(
    q,
    r,
    k,
    v,
    o,
    scale,
    cache_start,
    Sq,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    USE_CACHE_START: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """Forward-only Parallax over a cached KV (prefill / chunked decode).

    The ``Sq`` query tokens are the last ``Sq`` positions of a length-``Skv``
    sequence, so query row ``i`` is at absolute position ``Skv - Sq + i`` and
    attends to keys ``[0, Skv - Sq + i]`` (causal), further restricted to the
    sliding window and to ``[cache_start, Skv)`` when set. One program owns a
    ``BT``-row query block; see ``naive_parallax`` for the output formula.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G                               # kv head shared by this q head (GQA)
    RCP_LN2: tl.constexpr = 1.4426950216

    # First valid key index (left-padding); 0 when the whole cache is valid.
    if USE_CACHE_START:
        kv_lo = tl.load(cache_start + i_b).to(tl.int32)
    else:
        kv_lo = 0

    q_off = i_t * BT
    kv_offset = Skv - Sq                          # absolute position of query row 0
    rows = q_off + tl.arange(0, BT)
    abs_q = (kv_offset + rows)[:, None]           # [BT, 1] absolute position per query row
    row_mask = (rows < Sq)[:, None]               # [BT, 1] real (non-padded) query rows

    # Restrict the key-block loop to what this query block can reach: up to the
    # causal diagonal (KV_END_BLOCK) and down to the window's left edge (else 0).
    max_abs = kv_offset + tl.minimum(Sq, q_off + BT) - 1
    KV_END_BLOCK = tl.cdiv(tl.minimum(Skv, max_abs + 1), BS)
    if WINDOW_SIZE_LEFT >= 0:
        leftmost = tl.maximum(kv_lo, kv_offset + q_off - WINDOW_SIZE_LEFT + 1)
    else:
        leftmost = kv_lo
    KV_START_BLOCK = leftmost // BS

    p_q = tl.make_block_ptr(q + (i_b * Sq * HQ + i_hq) * K, (Sq, K), (HQ * K, 1), (q_off, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (i_b * Sq * HQ + i_hq) * K, (Sq, K), (HQ * K, 1), (q_off, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (i_b * Skv * H + i_h) * K, (Skv, K), (H * K, 1), (KV_START_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (i_b * Skv * H + i_h) * K, (Skv, K), (H * K, 1), (KV_START_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (i_b * Sq * HQ + i_hq) * K, (Sq, K), (HQ * K, 1), (q_off, 0), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    m_acc = tl.zeros((BT, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    Rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    for col_block_id in range(KV_START_BLOCK, KV_END_BLOCK):
        col = (col_block_id * BS + tl.arange(0, BS))[None, :]   # [1, BS] key positions
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")   # [BS, BK]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")   # [BS, BK]
        # [BT, BS]: causal, in-cache, past left-padding; optionally inside the window.
        mask = (abs_q >= col) & row_mask & (col < Skv) & (col >= kv_lo)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (col >= abs_q - WINDOW_SIZE_LEFT + 1)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2   # [BT, BS], base-2 logits
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        # finite pivot so a row with no valid key yet doesn't hit exp2(-inf - -inf) = NaN
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)                           # online-softmax rescale of running state
        w = exp2(qk - safe_m)                                  # [BT, BS] = p1 (unnormalized softmax)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)    # [BT, BS] = r @ k^T (unscaled)
        wr = w * rk                                            # [BT, BS] = p2 (unnormalized)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)    # running sum(p1)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)   # running sum(p2)
        barv_acc = alpha * barv_acc                            # running p1 @ v
        Rv_acc = alpha * Rv_acc                                # running p2 @ v
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Rows that see no valid key (e.g. left-padded query positions) have d1 == 0;
    # emit a finite zero instead of inf/NaN so padding can't poison valid rows.
    inv_d1 = tl.where(row_mask & (d1_acc > 0.0), 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1                                 # O1 / d1
    b_bart = d2_acc * inv_d1                                   # d2 / d1
    b_o = b_barv + b_bart * b_barv - Rv_acc * inv_d1           # O1/d1 * (1 + d2/d1) - O2/d1

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def parallax_decode(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cache_start: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Forward-only Parallax decode/prefill over a cached KV (inference; no autograd).

    The ``Sq`` query tokens are treated as the *last* ``Sq`` positions of a
    length-``Skv`` sequence: query ``i`` sits at absolute position ``Skv - Sq + i``
    and attends causally to keys ``[0, Skv - Sq + i]`` (and, with `window_size`,
    only the most recent `window_size` of them). `Sq == Skv` reduces to a full
    causal prefill; `Sq == 1` is a single decode step.

    Args:
        q (torch.Tensor):
            new queries of shape `[B, Sq, HQ, D]`.
        r (torch.Tensor):
            new secondary queries of shape `[B, Sq, HQ, D]`.
        k (torch.Tensor):
            cached keys of shape `[B, Skv, H, D]` (`Skv >= Sq`).
        v (torch.Tensor):
            cached values of shape `[B, Skv, H, D]`.
        scale (float, Optional):
            Scale applied to `q @ k^T` only. Defaults to `1 / sqrt(D)`. Default: `None`.
        window_size (int, Optional):
            Sliding-window length. Default: `None`.
        cache_start (torch.LongTensor, Optional):
            Per-batch first valid key index of shape `[B]` (to mask left-padding
            in the cache). `None` means the whole cache is valid. Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, Sq, HQ, D]`.
    """
    B, Sq, HQ, K = q.shape
    Skv, H = k.shape[1], k.shape[2]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5
    window_size_left = -1 if window_size is None else window_size

    q, r, k, v = (x.contiguous() for x in (q, r, k, v))
    if cache_start is not None:
        cache_start = cache_start.to(device=q.device, dtype=torch.int32).contiguous()
    BK = triton.next_power_of_2(K)
    BT = _block_size(K, q.device.index)
    o = torch.empty_like(q)
    grid = (triton.cdiv(Sq, BT), B * HQ)
    parallax_decode_kernel[grid](
        q, r, k, v, o, float(scale), cache_start, Sq, Skv,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        USE_CACHE_START=cache_start is not None,
        BT=BT, BS=BT,
        num_warps=8, num_stages=2,
    )
    return o


@triton.heuristics({
    'USE_CACHE_START': lambda args: args['cache_start'] is not None,
})
@triton.jit(do_not_specialize=['Skv'])
def parallax_decode_one_step_kernel(
    q,
    r,
    k,
    v,
    o,
    scale,
    cache_start,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    USE_CACHE_START: tl.constexpr,
    BS: tl.constexpr,
):
    """Single-token Parallax decode: one query per (batch, head) over its cached KV.

    The query is the sequence's last position, so causality is implicit (it sees
    every key) and only the lower bound from the window / left-padding matters.
    The query is held as a *vector* and reduced against the cache with an online
    softmax, so there is none of the wasted-tile compute the prefill-shaped
    ``parallax_decode_kernel`` incurs at ``Sq == 1``. One program per
    (batch, head); see ``naive_parallax`` for the output formula.
    """
    i_bh = tl.program_id(0)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G                               # kv head shared by this q head (GQA)
    RCP_LN2: tl.constexpr = 1.4426950216

    # Lowest key the query attends to: left-padding start, raised to the window edge.
    kv_lo = 0
    if USE_CACHE_START:
        kv_lo = tl.load(cache_start + i_b).to(tl.int32)
    if WINDOW_SIZE_LEFT >= 0:
        kv_lo = tl.maximum(kv_lo, Skv - WINDOW_SIZE_LEFT)
    kv_lo = tl.maximum(kv_lo, 0)

    p_q = tl.make_block_ptr(q + i_bh * K, (K,), (1,), (0,), (BK,), (0,))
    p_r = tl.make_block_ptr(r + i_bh * K, (K,), (1,), (0,), (BK,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * K, (K,), (1,), (0,), (BK,), (0,))
    b_q = tl.load(p_q, boundary_check=(0,), padding_option="zero").to(tl.float32)   # [BK] query vector
    b_r = tl.load(p_r, boundary_check=(0,), padding_option="zero").to(tl.float32)   # [BK] secondary query
    scale_log2 = scale * RCP_LN2

    # Running online-softmax state for the single query: pivot m, denominators
    # d1/d2 = sum(p1)/sum(p2), and unnormalized outputs o1/o2 = p1@v / p2@v.
    m = tl.full((1,), -float("inf"), dtype=tl.float32)
    d1 = tl.zeros((1,), dtype=tl.float32)
    d2 = tl.zeros((1,), dtype=tl.float32)
    o1 = tl.zeros((BK,), dtype=tl.float32)
    o2 = tl.zeros((BK,), dtype=tl.float32)

    start_block = kv_lo // BS
    p_k = tl.make_block_ptr(k + (i_b * Skv * H + i_h) * K, (Skv, K), (H * K, 1), (start_block * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (i_b * Skv * H + i_h) * K, (Skv, K), (H * K, 1), (start_block * BS, 0), (BS, BK), (1, 0))
    for i_s in range(start_block * BS, tl.cdiv(Skv, BS) * BS, BS):
        col = i_s + tl.arange(0, BS)
        mask = (col >= kv_lo) & (col < Skv)                          # [BS] valid keys
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")   # [BS, BK]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")   # [BS, BK]
        s1 = tl.sum(b_q[None, :] * b_k, axis=1) * scale_log2          # [BS] = scale * (q . k), base-2
        s2 = tl.sum(b_r[None, :] * b_k, axis=1)                        # [BS] = r . k (unscaled)
        s1 = tl.where(mask, s1, -float("inf"))
        m_new = tl.maximum(m, tl.max(s1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)         # finite pivot (empty cache -> 0)
        alpha = exp2(m - m_safe)                                      # rescale running state
        p1 = exp2(s1 - m_safe)                                         # [BS]
        p2 = p1 * s2                                                   # [BS]
        d1 = d1 * alpha + tl.sum(p1)
        d2 = d2 * alpha + tl.sum(p2)
        o1 = o1 * alpha + tl.sum(p1[:, None] * b_v, axis=0)           # [BK]
        o2 = o2 * alpha + tl.sum(p2[:, None] * b_v, axis=0)
        m = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    inv_d1 = tl.where(d1 > 0.0, 1.0 / d1, 0.0)                        # 0 when no valid key (avoid NaN)
    out = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1             # [BK] O1/d1*(1 + d2/d1) - O2/d1
    tl.store(p_o, out.to(p_o.dtype.element_ty), boundary_check=(0,))


def parallax_decode_one_step(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cache_start: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Single-token Parallax decode (one query per sequence over the cached KV).

    Optimized for the `Sq == 1` autoregressive step: the query is loaded as a
    vector (not a `[BT, D]` tile) and reduced against the cache with an online
    softmax, avoiding the wasted-tile compute of the prefill-shaped
    :func:`parallax_decode`. Forward-only (inference).

    Args:
        q (torch.Tensor):
            new query of shape `[B, 1, HQ, D]`.
        r (torch.Tensor):
            new secondary query of shape `[B, 1, HQ, D]`.
        k (torch.Tensor):
            cached keys of shape `[B, Skv, H, D]`.
        v (torch.Tensor):
            cached values of shape `[B, Skv, H, D]`.
        scale (float, Optional):
            Scale applied to `q @ k^T` only. Defaults to `1 / sqrt(D)`. Default: `None`.
        window_size (int, Optional):
            Sliding-window length; the query attends to the most recent `window_size` keys.
            Default: `None`.
        cache_start (torch.LongTensor, Optional):
            Per-batch first valid key index of shape `[B]` (left-padding). Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, 1, HQ, D]`.
    """
    B, Sq, HQ, K = q.shape
    if Sq != 1:
        raise ValueError(f"parallax_decode_one_step expects a single query (Sq=1), got Sq={Sq}")
    Skv, H = k.shape[1], k.shape[2]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5
    window_size_left = -1 if window_size is None else window_size

    q, r, k, v = (x.contiguous() for x in (q, r, k, v))
    if cache_start is not None:
        cache_start = cache_start.to(device=q.device, dtype=torch.int32).contiguous()
    BK = triton.next_power_of_2(K)
    o = torch.empty_like(q)
    grid = (B * HQ,)
    parallax_decode_one_step_kernel[grid](
        q, r, k, v, o, float(scale), cache_start, Skv,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        USE_CACHE_START=cache_start is not None,
        BS=128,
        num_warps=4, num_stages=2,
    )
    return o
