# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend backward for the GKA ridge solve.

The stock intra kernel keeps several `[BT, BK]` and `[BK, BK]` tiles live and overflows UB once
`BK = 128`. Gram and the lambda path share one launch so the `[BT, BT]` Gram stays in UB
instead of round-tripping through an extra kernel. State and the backward chunk scan stay
separate: state does not read that Gram, and the scan is serial across chunks. Each streams
K in panels of 64. A 32-wide panel issued 16 dots per nested pass and left those kernels scalar-bound.
Signs match `chunk_gka_solve_bwd`: the intra `dk` contributions are negated as they are stored,
and `dg` / `dh0` are flipped on the host.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.gka.chunk_solve_bwd import chunk_gka_solve_bwd
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM

_BK = 128
_BS = 64


@triton.jit
def _chunk_position(
    i_task, cu_seqlens, chunk_indices, T, BH, NT,
    H: tl.constexpr, BT: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_outer = i_task // BH
    i_inner = i_task % BH
    i_h = i_inner % H
    if IS_VARLEN:
        i_tg = i_outer
        i_b = tl.load(chunk_indices + i_tg * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_b).to(tl.int64)
        eos = tl.load(cu_seqlens + i_b + 1).to(tl.int64)
        seq_len = eos - bos
    else:
        i_b = i_inner // H
        i_t = i_outer
        i_tg = tl.cast(i_b, tl.int64) * NT + i_t
        bos = tl.cast(i_b, tl.int64) * T
        seq_len = tl.cast(T, tl.int64)
    n_tokens = tl.minimum(seq_len - tl.cast(i_t, tl.int64) * BT, BT)
    origin = bos + tl.cast(i_t, tl.int64) * BT
    return i_tg, i_h, i_b, bos, origin, n_tokens


@triton.jit
def _token_off(origin, i_h, cols, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr):
    token = origin + tl.arange(0, BT)
    return token[:, None] * (tl.cast(H, tl.int64) * K) + tl.cast(i_h, tl.int64) * K + cols[None, :]


@triton.jit
def _load_slab(ptr, origin, i_h, cols, n_tokens, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr):
    rows = tl.arange(0, BT)
    mask = (rows < n_tokens)[:, None] & (cols[None, :] < K)
    return tl.load(ptr + _token_off(origin, i_h, cols, H, K, BT), mask=mask, other=0.0).to(tl.float32)


@triton.jit
def _add_slab(ptr, origin, i_h, cols, n_tokens, value, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr):
    rows = tl.arange(0, BT)
    mask = (rows < n_tokens)[:, None] & (cols[None, :] < K)
    off = _token_off(origin, i_h, cols, H, K, BT)
    prev = tl.load(ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(ptr + off, prev + value, mask=mask)


@triton.jit
def _load_gate(
    gk, origin, bos, i_b, i_h, n_tokens, T,
    H: tl.constexpr, BT: tl.constexpr, IS_VARLEN: tl.constexpr, USE_GK: tl.constexpr,
):
    rows = tl.arange(0, BT)
    if USE_GK:
        token = origin + rows
        if IS_VARLEN:
            off = tl.cast(i_h, tl.int64) * T + token
        else:
            off = (tl.cast(i_b, tl.int64) * H + i_h) * T + (token - bos)
        return tl.load(gk + off, mask=rows < n_tokens, other=0.0).to(tl.float32)
    return tl.zeros([BT], dtype=tl.float32)


@triton.jit
def _gate_scalar(gk, token, bos, i_b, i_h, T, H: tl.constexpr, IS_VARLEN: tl.constexpr):
    if IS_VARLEN:
        off = tl.cast(i_h, tl.int64) * T + token
    else:
        off = (tl.cast(i_b, tl.int64) * H + i_h) * T + (token - bos)
    return tl.load(gk + off).to(tl.float32)


@triton.jit
def _add_dg(dg, origin, i_h, n_tokens, value, H: tl.constexpr, BT: tl.constexpr):
    rows = tl.arange(0, BT)
    mask = rows < n_tokens
    off = (origin + rows) * H + i_h
    prev = tl.load(dg + off, mask=mask, other=0.0)
    tl.store(dg + off, prev + value, mask=mask)


@triton.jit
def _causal_m(g, n_tokens, BT: tl.constexpr, USE_GK: tl.constexpr):
    rows = tl.arange(0, BT)
    valid = rows < n_tokens
    causal = valid[:, None] & valid[None, :] & (rows[:, None] <= rows[None, :])
    if USE_GK:
        return tl.where(causal, exp2(g[None, :] - g[:, None]), 0.0)
    return tl.where(causal, 1.0, 0.0)


@triton.jit
def _sq_panel(ptr, rows, cols, K):
    mask = (rows < K)[:, None] & (cols[None, :] < K)
    return tl.load(ptr + rows[:, None] * K + cols[None, :], mask=mask, other=0.0).to(tl.float32)


@triton.jit
def _store_panel(ptr, rows, cols, value, K):
    mask = (rows < K)[:, None] & (cols[None, :] < K)
    tl.store(ptr + rows[:, None] * K + cols[None, :], value, mask=mask)


@triton.jit(do_not_specialize=['T', 'task_offset', 'BH', 'NT', 'ridge_ratio'])
def _gka_bwd_gram_lamb_kernel(
    k, x, dq, h, fro, gk, dk, dg, hs, cu_seqlens, chunk_indices,
    ridge_ratio, T, task_offset, BH, NT,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BS: tl.constexpr,
    USE_GK: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_task = tl.cast(tl.program_id(0), tl.int64) + task_offset
    i_tg, i_h, i_b, bos, origin, n_tokens = _chunk_position(
        i_task, cu_seqlens, chunk_indices, T, BH, NT, H, BT, IS_VARLEN,
    )
    rows = tl.arange(0, BT)
    g = _load_gate(gk, origin, bos, i_b, i_h, n_tokens, T, H, BT, IS_VARLEN, USE_GK)
    m = _causal_m(g, n_tokens, BT, USE_GK)
    if USE_GK:
        gh = tl.where(rows < n_tokens, exp2(g), 0.0)
    else:
        gh = tl.where(rows < n_tokens, 1.0, 0.0)
    gram_kk = tl.zeros([BT, BT], dtype=tl.float32)
    gram_kx = tl.zeros([BT, BT], dtype=tl.float32)
    gram_kd = tl.zeros([BT, BT], dtype=tl.float32)
    for _s0 in tl.static_range(BK // BS):
        cols = _s0 * BS + tl.arange(0, BS)
        # A tl.dot left operand is not safe to reread, so each product reloads its tiles.
        gram_kk += tl.dot(
            _load_slab(k, origin, i_h, cols, n_tokens, H, K, BT),
            tl.trans(_load_slab(k, origin, i_h, cols, n_tokens, H, K, BT)),
        )
        gram_kx += tl.dot(
            _load_slab(k, origin, i_h, cols, n_tokens, H, K, BT),
            tl.trans(_load_slab(x, origin, i_h, cols, n_tokens, H, K, BT)),
        )
        gram_kd += tl.dot(
            _load_slab(k, origin, i_h, cols, n_tokens, H, K, BT),
            tl.trans(_load_slab(dq, origin, i_h, cols, n_tokens, H, K, BT)),
        )
    s = gram_kx * m
    dg_gram = tl.zeros([BT], dtype=tl.float32)
    if USE_GK:
        causal = (rows[:, None] <= rows[None, :]) & (rows[:, None] < n_tokens) & (rows[None, :] < n_tokens)
        dm = tl.where(causal, s * gram_kd, 0.0)
        dg_gram = tl.sum(dm, 0) - tl.sum(dm, 1)
    for _s0 in tl.static_range(BK // BS):
        cols = _s0 * BS + tl.arange(0, BS)
        acc = tl.dot(s + 0.0, _load_slab(dq, origin, i_h, cols, n_tokens, H, K, BT))
        acc += tl.dot(gram_kd * m, _load_slab(x, origin, i_h, cols, n_tokens, H, K, BT))
        _add_slab(dk, origin, i_h, cols, n_tokens, -acc, H, K, BT)

    # kx / kd are dead. kk stays live and feeds the lambda path, so it is not stored.
    b_fro = tl.load(fro + (origin + rows) * H + i_h, mask=rows < n_tokens, other=1.0).to(tl.float32)
    qx = tl.zeros([BT], dtype=tl.float32)
    for _s0 in tl.static_range(BK // BS):
        cols = _s0 * BS + tl.arange(0, BS)
        qx += tl.sum(
            _load_slab(dq, origin, i_h, cols, n_tokens, H, K, BT)
            * _load_slab(x, origin, i_h, cols, n_tokens, H, K, BT),
            1,
        )
    tmp = ridge_ratio * qx
    w = tl.where(rows < n_tokens, 2.0 * tmp / b_fro, 0.0)
    w1 = tl.sum(m * (gh * w)[None, :], 1)
    w2 = tl.dot(m * w[None, :], tl.trans(m + 0.0))
    p_h = h + (i_tg * H + i_h) * K * K
    p_hs = hs + (i_tg * H + i_h) * K * K
    dg_t = tl.zeros([BT], dtype=tl.float32)
    for _pc in tl.static_range(BK // BS):
        cols = _pc * BS + tl.arange(0, BS)
        k_pc = _load_slab(k, origin, i_h, cols, n_tokens, H, K, BT)
        acc = tl.dot(gram_kk * w2, k_pc)
        for _pr in tl.static_range(BK // BS):
            rws = _pr * BS + tl.arange(0, BS)
            acc += w1[:, None] * tl.dot(
                _load_slab(k, origin, i_h, rws, n_tokens, H, K, BT),
                _sq_panel(p_h, rws, cols, K),
            )
        _add_slab(dk, origin, i_h, cols, n_tokens, -acc, H, K, BT)
        dg_t += tl.sum(acc * k_pc, 1)
    scale = tl.sum(gh * gh * w)
    for _pr in tl.static_range(BK // BS):
        rws = _pr * BS + tl.arange(0, BS)
        for _pc in tl.static_range(BK // BS):
            cols = _pc * BS + tl.arange(0, BS)
            # Reload k: the transposed tile is a tl.dot left operand and must not be reused.
            k_pr = _load_slab(k, origin, i_h, rws, n_tokens, H, K, BT)
            block = tl.dot(tl.trans(k_pr * w1[:, None]), _load_slab(k, origin, i_h, cols, n_tokens, H, K, BT))
            block += scale * _sq_panel(p_h, rws, cols, K)
            _store_panel(p_hs, rws, cols, block, K)
    if USE_GK:
        _add_dg(dg, origin, i_h, n_tokens, dg_gram + tmp * b_fro - 0.5 * dg_t, H, BT)


@triton.jit(do_not_specialize=['T', 'task_offset', 'BH', 'NT'])
def _gka_bwd_state_kernel(
    k, x, dq, h, dh, gk, dk, dg, cu_seqlens, chunk_indices,
    T, task_offset, BH, NT,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BS: tl.constexpr,
    USE_GK: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_task = tl.cast(tl.program_id(0), tl.int64) + task_offset
    i_tg, i_h, i_b, bos, origin, n_tokens = _chunk_position(
        i_task, cu_seqlens, chunk_indices, T, BH, NT, H, BT, IS_VARLEN,
    )
    rows_t = tl.arange(0, BT)
    g = _load_gate(gk, origin, bos, i_b, i_h, n_tokens, T, H, BT, IS_VARLEN, USE_GK)
    if USE_GK:
        g_last = _gate_scalar(gk, origin + n_tokens - 1, bos, i_b, i_h, T, H, IS_VARLEN)
        gh = tl.where(rows_t < n_tokens, exp2(g), 0.0)
        gl = tl.where(rows_t < n_tokens, exp2(g_last - g), 0.0)
    else:
        g_last = 0.0
        gh = tl.zeros([BT], dtype=tl.float32)
        gl = tl.where(rows_t < n_tokens, 1.0, 0.0)
    state = (i_tg * H + i_h) * K * K
    p_h = h + state
    p_dh = dh + state

    for _pc in tl.static_range(BK // BS):
        cols = _pc * BS + tl.arange(0, BS)
        acc = tl.zeros([BT, BS], dtype=tl.float32)
        for _pr in tl.static_range(BK // BS):
            rws = _pr * BS + tl.arange(0, BS)
            k2 = _load_slab(k, origin, i_h, rws, n_tokens, H, K, BT) * gl[:, None]
            acc += tl.dot(k2, _sq_panel(p_dh, rws, cols, K) + tl.trans(_sq_panel(p_dh, cols, rws, K)))
        _add_slab(dk, origin, i_h, cols, n_tokens, -acc, H, K, BT)

    if USE_GK:
        dgl = tl.zeros([BT], dtype=tl.float32)
        for _pc in tl.static_range(BK // BS):
            cols = _pc * BS + tl.arange(0, BS)
            y = tl.zeros([BT, BS], dtype=tl.float32)
            for _pr in tl.static_range(BK // BS):
                rws = _pr * BS + tl.arange(0, BS)
                k2 = _load_slab(k, origin, i_h, rws, n_tokens, H, K, BT) * gl[:, None]
                y += tl.dot(k2, _sq_panel(p_dh, rws, cols, K))
            dgl += tl.sum(y * _load_slab(k, origin, i_h, cols, n_tokens, H, K, BT), 1)
        hx = tl.zeros([BT], dtype=tl.float32)
        fro_dh = 0.0
        for _pr in tl.static_range(BK // BS):
            rws = _pr * BS + tl.arange(0, BS)
            partial = tl.zeros([BT, BS], dtype=tl.float32)
            for _pc in tl.static_range(BK // BS):
                cols = _pc * BS + tl.arange(0, BS)
                h_block = _sq_panel(p_h, rws, cols, K)
                fro_dh += tl.sum(h_block * _sq_panel(p_dh, rws, cols, K))
                partial += tl.dot(
                    _load_slab(x, origin, i_h, cols, n_tokens, H, K, BT),
                    tl.trans(h_block + 0.0),
                )
            hx += tl.sum(partial * _load_slab(dq, origin, i_h, rws, n_tokens, H, K, BT), 1)
        dg_vec = hx * gh - dgl
        dg_vec = tl.where(rows_t == n_tokens - 1, dg_vec + tl.sum(dgl) + fro_dh * exp2(g_last), dg_vec)
        _add_dg(dg, origin, i_h, n_tokens, dg_vec, H, BT)


@triton.jit(do_not_specialize=['T', 'task_offset'])
def _gka_bwd_inter_kernel(
    k, h, hs, gk, dk, dg, carry, cu_seqlens, chunk_offsets,
    T, task_offset,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BS: tl.constexpr,
    USE_GK: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_nh = tl.cast(tl.program_id(0), tl.int64) + task_offset
    i_n = i_nh // H
    i_h = i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos = i_n * tl.cast(T, tl.int64)
        seq_len = tl.cast(T, tl.int64)
        boh = i_n * tl.cdiv(T, BT)
    n_chunks = tl.cdiv(seq_len, BT)
    p_carry = carry + i_nh * K * K
    for i_c in range(n_chunks - 1, -1, -1):
        i_t = tl.cast(i_c, tl.int64)
        origin = bos + i_t * BT
        n_tokens = tl.minimum(seq_len - i_t * BT, BT)
        rows_t = tl.arange(0, BT)
        if USE_GK:
            g = _load_gate(gk, origin, bos, i_n, i_h, n_tokens, T, H, BT, IS_VARLEN, True)
            g_last = _gate_scalar(gk, origin + n_tokens - 1, bos, i_n, i_h, T, H, IS_VARLEN)
            gl = tl.where(rows_t < n_tokens, exp2(g_last - g), 0.0)
        else:
            g_last = 0.0
            gl = tl.where(rows_t < n_tokens, 1.0, 0.0)
        b_dg = tl.zeros([BT], dtype=tl.float32)
        for _pc in tl.static_range(BK // BS):
            cols = _pc * BS + tl.arange(0, BS)
            acc = tl.zeros([BT, BS], dtype=tl.float32)
            for _pr in tl.static_range(BK // BS):
                rws = _pr * BS + tl.arange(0, BS)
                k2 = _load_slab(k, origin, i_h, rws, n_tokens, H, K, BT) * gl[:, None]
                acc += tl.dot(k2, _sq_panel(p_carry, rws, cols, K))
            contrib = -acc
            k_pc = _load_slab(k, origin, i_h, cols, n_tokens, H, K, BT)
            b_dg += tl.sum(contrib * k_pc, 1)
            _add_slab(dk, origin, i_h, cols, n_tokens, contrib, H, K, BT)
        b_dg *= 0.5
        state = ((boh + i_t) * H + i_h) * K * K
        if USE_GK:
            gamma = exp2(g_last)
        else:
            gamma = 1.0
        inner = 0.0
        for _pr in tl.static_range(BK // BS):
            rws = _pr * BS + tl.arange(0, BS)
            for _pc in tl.static_range(BK // BS):
                cols = _pc * BS + tl.arange(0, BS)
                panel = _sq_panel(p_carry, rws, cols, K) * gamma
                if USE_GK:
                    inner += tl.sum(panel * _sq_panel(h + state, rws, cols, K))
                panel += _sq_panel(hs + state, rws, cols, K)
                _store_panel(p_carry, rws, cols, panel, K)
        if USE_GK:
            b_dg = tl.where(rows_t == n_tokens - 1, b_dg + inner * 0.5 - tl.sum(b_dg), b_dg)
            _add_dg(dg, origin, i_h, n_tokens, b_dg, H, BT)


def _launch_chunks(kernel, task_num: int, kwargs: dict) -> None:
    if task_num <= 0:
        return
    # workspace multibuffer adds a CV slot that pushes the 4-panel state over the 192KiB UB.
    compile_kwargs = ascend_compile_kwargs(workspace_multibuffer=0)
    for offset in range(0, task_num, ASCEND_MAX_GRID_DIM):
        count = min(ASCEND_MAX_GRID_DIM, task_num - offset)
        kernel[(count,)](task_offset=offset, **kwargs, **compile_kwargs)


def chunk_gka_solve_bwd_npu(
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
    # Ascend has one staged implementation; split controls only the stock kernel's launch policy.
    del split
    B, T, H, K = k.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
        BH = B * H
        n_chunks = NT
    else:
        N, n_chunks = len(cu_seqlens) - 1, len(chunk_indices)
        NT = n_chunks
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        BH = H
    use_gk = gk is not None
    # [B, H, T] so repeated loads along time are contiguous. H == 1 does not copy.
    gk_c = None if gk is None else gk.transpose(1, 2)
    if gk_c is not None and not gk_c.is_contiguous():
        gk_c = gk_c.contiguous()
    dk = torch.zeros(B, T, H, K, dtype=torch.float32, device=k.device)
    dg = torch.zeros(B, T, H, dtype=torch.float32, device=k.device) if use_gk else None
    hs = torch.empty_like(h, dtype=torch.float32)
    carry = k.new_empty(N, H, K, K, dtype=torch.float32) if output_dh0 else torch.empty(
        N, H, K, K, dtype=torch.float32, device=k.device,
    )
    if output_dh0:
        dh0 = carry
    else:
        dh0 = None
    carry.zero_()
    common = dict(
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, BH=BH, NT=NT,
        H=H, K=K, BT=BT, BK=_BK, BS=_BS, USE_GK=use_gk, IS_VARLEN=cu_seqlens is not None,
    )
    tasks = n_chunks * BH
    _launch_chunks(_gka_bwd_gram_lamb_kernel, tasks, dict(
        k=k, x=x, dq=dq, h=h, fro=fro, gk=gk_c, dk=dk, dg=dg, hs=hs,
        ridge_ratio=ridge_ratio, **common,
    ))
    _launch_chunks(_gka_bwd_state_kernel, tasks, dict(
        k=k, x=x, dq=dq, h=h, dh=dh, gk=gk_c, dk=dk, dg=dg, **common,
    ))
    _launch_chunks(_gka_bwd_inter_kernel, N * H, dict(
        k=k, h=h, hs=hs, gk=gk_c, dk=dk, dg=dg, carry=carry,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets, T=T,
        H=H, K=K, BT=BT, BK=_BK, BS=_BS, USE_GK=use_gk, IS_VARLEN=cu_seqlens is not None,
    ))
    if dg is not None:
        dg = -dg
    if dh0 is not None:
        dh0 = -0.5 * dh0
    return dk.to(dtype=k.dtype), dg, dh0


def chunk_gka_solve_bwd_verifier(
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
) -> tuple[bool, str | None]:
    del x, dq, h, dh, fro, gk, ridge_ratio, output_dh0, cu_seqlens, chunk_indices, split
    if k.device.type != 'npu':
        return False, 'GKA Ascend backward requires NPU tensors'
    if k.shape[-1] > 128:
        return False, f'GKA Ascend backward supports K <= 128, got {k.shape[-1]}'
    if chunk_size > 64:
        return False, f'GKA Ascend backward requires chunk_size <= 64, got {chunk_size}'
    if k.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False, 'GKA Ascend backward received an unsupported dtype'
    return True, None


@register(chunk_gka_solve_bwd, backend=TritonAscendBackend, verifier=chunk_gka_solve_bwd_verifier)
def chunk_gka_solve_bwd_npu_registered(
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
):
    return chunk_gka_solve_bwd_npu(
        k, x, dq, h, dh, fro, gk=gk, ridge_ratio=ridge_ratio, output_dh0=output_dh0,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices, split=split,
    )
