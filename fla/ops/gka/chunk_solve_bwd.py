# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
# PART is explained under `split` in `chunk_gka_solve_bwd`. PART=2 adds into `dk` and `dg`, so autotuning restores them
# between trials. The autotune result is not cached on disk, so that every process benchmarks configs on its own GPU
# and skips those that do not fit.
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2]
    ],
    key=['H', 'K', 'BT', 'PART'],
    restore_value=['dk', 'dg'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_gka_bwd_kernel_dk_intra(
    k,
    x,
    dq,
    h,
    dh,
    fro,
    gk,
    dk,
    dg,
    hs,
    cu_seqlens,
    chunk_indices,
    ridge_ratio,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    PART: tl.constexpr,
    USE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # `PART != 2` guards the terms that need only `dh`, and `PART != 1` guards the terms that need `h`
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    x += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    fro += bos * H + i_h
    # `h` is `H_{[c]}` for chunk `i_t`, and `dh` the gradient w.r.t. the state at its end
    h += (i_tg * H + i_h) * K*K
    dh += (i_tg * H + i_h) * K*K
    hs += (i_tg * H + i_h) * K*K

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]
    m_kk = m_k[:, None] & m_k[None, :]
    # index of the last valid row of this chunk
    i_last = tl.minimum(BT, T - i_t * BT) - 1

    # [BT, BK]
    b_k = tl.load(k + o_t[:, None] * (H*K) + o_k[None, :], mask=m_tk, other=0.)
    b_x = tl.load(x + o_t[:, None] * (H*K) + o_k[None, :], mask=m_tk, other=0.)
    b_dq = tl.load(dq + o_t[:, None] * (H*K) + o_k[None, :], mask=m_tk, other=0.)
    # [BK, BK]
    if PART != 1:
        b_h = tl.load(h + o_k[:, None] * K + o_k[None, :], mask=m_kk, other=0.)
    if PART != 2 or USE_GK:
        b_dh = tl.load(dh + o_k[:, None] * K + o_k[None, :], mask=m_kk, other=0.)

    # [BT, BT] indexed [j, t]: token j feeds `H_t` for every t >= j
    m_s = (o_t[:, None] <= o_t[None, :]) & m_t[:, None] & m_t[None, :]
    if USE_GK:
        b_g = tl.load(gk + (bos * H + i_h) + o_t * H, mask=m_t, other=0.).to(tl.float32)
        b_g_last = tl.load(gk + (bos * H + i_h) + (i_t * BT + i_last) * H).to(tl.float32)
        b_m = tl.where(m_s, exp2(b_g[None, :] - b_g[:, None]), 0)
        b_gh = exp2(b_g)
        b_gl = exp2(b_g_last - b_g)
    else:
        b_m = m_s.to(tl.float32)
        b_gh = tl.full([BT], 1., dtype=tl.float32)
        b_gl = tl.full([BT], 1., dtype=tl.float32)

    # direct path through H_t
    if PART != 2:
        b_s = tl.dot(b_k, tl.trans(b_x)) * b_m
        b_ds = tl.dot(b_k, tl.trans(b_dq))
        b_k2 = (b_k * b_gl[:, None]).to(b_k.dtype)
        b_dk = tl.dot(b_k2, (b_dh + tl.trans(b_dh)).to(b_k.dtype))
        b_dk += tl.dot(b_s.to(b_dq.dtype), b_dq)
        b_dk += tl.dot((b_ds * b_m).to(b_x.dtype), b_x)
    else:
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)

    if USE_GK:
        b_dg = tl.zeros([BT], dtype=tl.float32)
        if PART != 2:
            b_dm = tl.where(m_s, b_s * b_ds, 0)
            b_dg = tl.sum(b_dm, axis=0) - tl.sum(b_dm, axis=1)
        if PART != 1:
            b_dg += tl.sum(tl.dot(b_dq, b_h) * b_gh[:, None] * b_x, axis=1)
        b_dg_last = 0.
        if PART != 2:
            b_dgl = tl.sum(tl.dot(b_k2, b_dh.to(b_k.dtype)) * b_k, axis=1)
            b_dg -= b_dgl
            b_dg_last = tl.sum(b_dgl)
        if PART != 1:
            b_dg_last += tl.sum(b_dh * b_h) * exp2(b_g_last)
        b_dg = tl.where(tl.arange(0, BT) == i_last, b_dg + b_dg_last, b_dg)

    # path through lamb_t = ridge_ratio * ||H_t||_F
    if PART != 1:
        b_fro = tl.load(fro + o_t * H, mask=m_t, other=1.)
        b_tmp = ridge_ratio * tl.sum(b_dq * b_x, axis=1)
        if USE_GK:
            b_dg += b_tmp * b_fro
        b_w = tl.where(m_t, 2 * b_tmp / b_fro, 0.)
        b_w1 = tl.sum(b_m * (b_gh * b_w)[None, :], axis=1)
        b_w2 = tl.dot(b_m * b_w[None, :], tl.trans(b_m))
        b_t = tl.dot((tl.dot(b_k, tl.trans(b_k)) * b_w2).to(b_k.dtype), b_k).to(tl.float32)
        b_t += tl.dot((b_w1[:, None] * b_k).to(b_k.dtype), b_h.to(b_k.dtype))
        b_dk += b_t
        if USE_GK:
            b_dg -= tl.sum(b_t * b_k, axis=1) * 0.5

        # this chunk's contribution to the lamb-path gradient w.r.t. its chunk-start state
        b_hs = tl.sum(b_gh * b_gh * b_w) * b_h
        b_hs += tl.dot(tl.trans((b_w1[:, None] * b_k).to(b_k.dtype)), b_k)
        tl.store(hs + o_k[:, None] * K + o_k[None, :], b_hs.to(hs.dtype.element_ty), mask=m_kk)

    p_dk = dk + o_t[:, None] * (H*K) + o_k[None, :]
    # `b_dk` is negated, like `dh`, so it is flipped when stored; PART=2 adds to what PART=1 stored
    if PART == 2:
        b_dk -= tl.load(p_dk, mask=m_tk, other=0.).to(tl.float32)
    tl.store(p_dk, (-b_dk).to(dk.dtype.element_ty), mask=m_tk)
    if USE_GK:
        p_dg = dg + (bos * H + i_h) + o_t * H
        if PART == 2:
            b_dg += tl.load(p_dg, mask=m_t, other=0.)
        tl.store(p_dg, b_dg.to(dg.dtype.element_ty), mask=m_t)


@triton.heuristics({
    'USE_GK': lambda args: args['gk'] is not None,
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
# `dk` and `dg` are updated in place, so autotuning must restore them after every benchmark run
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3]
    ],
    key=['H', 'K', 'BT'],
    restore_value=['dk', 'dg'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gka_bwd_kernel_dk_inter(
    k,
    h,
    hs,
    gk,
    dk,
    dg,
    dh0,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_GK: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    k += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_kk = m_k[:, None] & m_k[None, :]

    # lamb-path gradient w.r.t. the state at the end of the current chunk, carried backwards over chunks
    b_hs = tl.zeros([BK, BK], dtype=tl.float32)
    for i_c in range(NT - 1, -1, -1):
        i_t = i_c.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tk = m_t[:, None] & m_k[None, :]
        i_last = tl.minimum(BT, T - i_t * BT) - 1
        o_s = ((boh + i_t) * H + i_h) * K*K

        b_k = tl.load(k + o_t[:, None] * (H*K) + o_k[None, :], mask=m_tk, other=0.)
        b_h = tl.load(h + o_s + o_k[:, None] * K + o_k[None, :], mask=m_kk, other=0.)
        b_hs_local = tl.load(hs + o_s + o_k[:, None] * K + o_k[None, :], mask=m_kk, other=0.)
        if USE_GK:
            b_g = tl.load(gk + (bos * H + i_h) + o_t * H, mask=m_t, other=0.).to(tl.float32)
            b_g_last = tl.load(gk + (bos * H + i_h) + (i_t * BT + i_last) * H).to(tl.float32)
            b_gl = exp2(b_g_last - b_g)
        else:
            b_gl = tl.full([BT], 1., dtype=tl.float32)

        # negated here, since it is added to the true-sign `dk` stored by the intra kernel
        b_dk = -tl.dot((b_k * b_gl[:, None]).to(b_k.dtype), b_hs.to(b_k.dtype))
        if USE_GK:
            b_dg = tl.sum(b_dk.to(tl.float32) * b_k, axis=1) * 0.5
            b_hs *= exp2(b_g_last)
            b_dg_last = tl.sum(b_hs * b_h) * 0.5 - tl.sum(b_dg)
            b_dg = tl.where(tl.arange(0, BT) == i_last, b_dg + b_dg_last, b_dg)
            p_dg = dg + (bos * H + i_h) + o_t * H
            tl.store(p_dg, (tl.load(p_dg, mask=m_t, other=0.) + b_dg).to(dg.dtype.element_ty), mask=m_t)
        b_hs += b_hs_local

        p_dk = dk + o_t[:, None] * (H*K) + o_k[None, :]
        tl.store(p_dk, (tl.load(p_dk, mask=m_tk, other=0.).to(tl.float32) + b_dk).to(dk.dtype.element_ty), mask=m_tk)

    if STORE_INITIAL_STATE_GRADIENT:
        tl.store(dh0 + i_nh * K*K + o_k[:, None] * K + o_k[None, :], b_hs.to(dh0.dtype.element_ty), mask=m_kk)


def chunk_gka_solve_bwd(
    k: torch.Tensor,
    x: torch.Tensor,
    dq: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    fro: torch.Tensor,
    gk: torch.Tensor | None = None,
    ridge_ratio: float = 0.02,
    output_dh0: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    split: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    r"""
    Gradients of the ridge solve `x_t = (H_t + ridge_ratio * ||H_t||_F * I)^{-1} q_t` w.r.t. the keys and the
    decay of `H_t`, given `dq_t = (H_t + lamb_t I)^{-1} dx_t` and the gradients w.r.t. `H_t` from `chunk_bwd_dh`.

    Args:
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        x (torch.Tensor):
            Forward solutions of shape `[B, T, H, K]`.
        dq (torch.Tensor):
            Adjoint solutions of shape `[B, T, H, K]`, i.e. the gradient w.r.t. the right-hand side `q`.
        h (torch.Tensor):
            `H_{[c]}`, the state at the start of each chunk, of shape `[B, NT, H, K, K]`, where `NT` is the number of
            chunks per sequence, or the total over all sequences with `cu_seqlens`.
        dh (torch.Tensor):
            `chunk_bwd_dh(q=dq, k=k, v=k, do=x, dht=-dht, ...)`: the negated gradients w.r.t. `H_{[c+1]}`,
            the state at the end of each chunk, of shape `[B, NT, H, K, K]`.
            The kernels return `dk` with the true sign, and `dg` and `dh0` are flipped afterwards in this function.
        fro (torch.Tensor):
            `||H_t||_F` of shape `[B, T, H]` from the forward solve.
        gk (torch.Tensor, Optional):
            Chunk-local cumulative log2-space decay of `H_t`, of shape `[B, T, H]`. Default: `None`.
        ridge_ratio (float, Optional):
            Sets the ridge strength `lamb_t = ridge_ratio * ||H_t||_F`. Default: `0.02`.
        output_dh0 (bool, Optional):
            Whether to return the gradient of the initial `h_kk` through `lamb`. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs. Default: `None`.
        chunk_size (int, Optional):
            Chunk size used in the forward pass. Default: `64`.
        chunk_indices (torch.LongTensor, Optional):
            Precomputed `prepare_chunk_indices(cu_seqlens, chunk_size)`. Default: `None`.
        split (bool, Optional):
            Whether to run the per-chunk kernel as two launches instead of one, with the same result.
            One launch (`PART=0`) computes the chunk's whole contribution to `dk` and `dg`.
            On H100/H200, it fits in shared memory for `bfloat16` and `float16` at every `K` up to 128,
            and for `float32` with `K <= 64`. For `float32` with `K > 64` it does not,
            so the work is split into the terms that need only `dh` (`PART=1`) and then those that need `h` (`PART=2`).
            If `None`, the split is used exactly for `float32` with `K > 64`. Default: `None`.

    Returns:
        dk (torch.Tensor):
            Gradient w.r.t. `k` through the solve, of shape `[B, T, H, K]`.
        dg (torch.Tensor | None):
            Gradient w.r.t. the chunk-local cumulative decay `gk` (before the reverse cumsum), of shape `[B, T, H]`
            in `float32`, or `None` if `gk` is `None`.
        dh0_lamb (torch.Tensor | None):
            The `lamb`-path part of the gradient w.r.t. the initial `h_kk`, of shape `[N, H, K, K]` in `float32`, or
            `None` if `output_dh0` is `False`. The full gradient adds `-dh0` from `chunk_bwd_dh`.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    BK = max(triton.next_power_of_2(K), 16)

    dk = torch.empty_like(k)
    dg = torch.empty(B, T, H, dtype=torch.float, device=k.device) if gk is not None else None
    hs = torch.empty_like(h)
    dh0 = k.new_empty(N, H, K, K, dtype=torch.float) if output_dh0 else None

    if split is None:
        # the single-launch kernel does not fit in shared memory for float32 with K > 64
        split = k.dtype == torch.float32 and K > 64
    for part in ((1, 2) if split else (0,)):
        chunk_gka_bwd_kernel_dk_intra[(NT, B * H)](
            k=k,
            x=x,
            dq=dq,
            h=h,
            dh=dh,
            fro=fro,
            gk=gk,
            dk=dk,
            dg=dg,
            hs=hs,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            ridge_ratio=ridge_ratio,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BK=BK,
            PART=part,
        )
    chunk_gka_bwd_kernel_dk_inter[(N * H,)](
        k=k,
        h=h,
        hs=hs,
        gk=gk,
        dk=dk,
        dg=dg,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
    )
    # the kernels leave `dg` negated, like `dh`, and `dh0` negated and doubled
    if dg is not None:
        dg = -dg
    if dh0 is not None:
        dh0 = -0.5 * dh0
    return dk, dg, dh0
