# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA chunk intra token-parallel kernels for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.common.backends.triton_ascend.chunk_delta_h import mload, mstore
from fla.ops.kda.chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs, input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    get_ub_manager,
    max_grid_axis_chunks,
)

HEAD_BLOCK_SIZE = 4
HEAD_TILES = (4, 8, 16, 32)
COLUMN_TILES = (1, 2, 4, 8, 16)


def prune_head_tiles(configs, nargs, **kwargs):
    args = {**nargs, **kwargs}
    # each column adds a full set of FP32 query, key, gate, and temporary rows.
    limits = {}
    for bj in COLUMN_TILES:
        limits[bj] = compute_row_tile_block_size(
            row_dim=max(HEAD_TILES),
            fixed_dim=triton.next_power_of_2(args['K']),
            memory_multiplier=8.0 * bj,
            safety_margin=0.75,
            fallback=1,
            min_block=1,
            max_block=max(HEAD_TILES),
        )
    return [config for config in configs
            if config.kwargs.get('BH') in HEAD_TILES and config.kwargs.get('BJ') in COLUMN_TILES
            and config.kwargs['BH'] <= limits[config.kwargs['BJ']]]


def launch_token_parallel_kernel(
    kernel,
    *,
    tg_total: int,
    hv_total: int,
    kernel_kwargs: dict,
) -> None:
    hg_total = triton.cdiv(hv_total, min(HEAD_TILES))
    max_tg = max_grid_axis_chunks(tg_total, hg_total, max_grid=ASCEND_MAX_GRID_DIM)
    for tg_off in range(0, tg_total, max_tg):
        tg_len = min(max_tg, tg_total - tg_off)
        kernel_kwargs['TG_OFFSET'] = tg_off
        max_hg = max_grid_axis_chunks(hg_total, tg_len, max_grid=ASCEND_MAX_GRID_DIM)
        head_step = max(max(HEAD_TILES), max_hg * min(HEAD_TILES) // max(HEAD_TILES) * max(HEAD_TILES))
        for head_off in range(0, hv_total, head_step):
            head_len = min(head_step, hv_total - head_off)
            kernel_kwargs['HEAD_OFFSET'] = head_off
            kernel_kwargs['TG_LEN'] = tg_len
            kernel_kwargs['HEAD_LEN'] = head_len
            kernel[lambda meta: (tg_len, triton.cdiv(head_len, meta['BH']))](**kernel_kwargs)


@triton.heuristics({
    'MASK_DMA': lambda args: not (
        args['HV'] % args['BH'] == 0 and triton.next_power_of_2(args['K']) == args['K']
        and (args.get('BJ', 1) == 1 or (
            not args['IS_VARLEN'] and args['T'] % args['BJ'] == 0 and args['BC'] % args['BJ'] == 0
        ))
    ),
})
@triton.jit(do_not_specialize=['T', 'N', 'TG_LEN', 'HEAD_LEN', 'UB_CAPACITY'])
def chunk_kda_fwd_kernel_intra_token_parallel_npu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    TG_LEN,
    HEAD_LEN,
    UB_CAPACITY,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TG_OFFSET: tl.constexpr,
    HEAD_OFFSET: tl.constexpr,
    MASK_DMA: tl.constexpr,
    BJ: tl.constexpr = 1,
):
    i_tg = tl.program_id(0) + TG_OFFSET
    i_hg = tl.program_id(1)

    if IS_VARLEN:
        # graph replay can leave unused token slots in the fixed grid;
        # skip them before sequence lookup, which would yield i_n=N and read cu_seqlens[N+1] out of bounds.
        actual_t = tl.load(cu_seqlens + N).to(tl.int64)
        if tl.cast(i_tg, tl.int64) >= actual_t:
            return
        i_n = 0
        left, right = 0, N

        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        i_t = i_tg - bos
    else:
        bos = tl.cast(i_tg // T, tl.int64) * T
        i_t = i_tg % T

    i_t = tl.cast(i_t, tl.int64)
    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    G: tl.constexpr = HV // H

    q += bos * H * K
    k += bos * H * K
    g += bos * HV * K
    Aqk += bos * HV * BT
    Akk += bos * HV * BC
    beta += bos * HV

    BK: tl.constexpr = triton.next_power_of_2(K)
    head_base = tl.cast(HEAD_OFFSET, tl.int64) + tl.cast(i_hg, tl.int64) * BH
    o_hv = head_base + tl.arange(0, BH)
    o_h = o_hv // G
    o_k = tl.arange(0, BK)
    m_hv = o_hv < HV
    m_k = o_k < K
    m_hk = m_hv[:, None] & m_k[None, :]

    p_qk = o_h[:, None] * K + o_k[None, :]
    b_q = mload(q + i_t * H * K + p_qk, m_hk, MASK_DMA).to(tl.float32)
    b_k = mload(k + i_t * H * K + p_qk, m_hk, MASK_DMA).to(tl.float32)
    p_g = o_hv[:, None] * K + o_k[None, :]
    b_g = mload(g + i_t * HV * K + p_g, m_hk, MASK_DMA).to(tl.float32)
    b_k *= mload(beta + i_t * HV + o_hv, m_hv, MASK_DMA).to(tl.float32)[:, None]

    if BJ == 1:
        for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
            j64 = tl.cast(j, tl.int64)
            b_kj = mload(k + j64 * H * K + p_qk, m_hk, MASK_DMA).to(tl.float32)
            b_gj = mload(g + j64 * HV * K + p_g, m_hk, MASK_DMA).to(tl.float32)
            b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
            b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
            b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)
            mstore(Aqk + i_t * HV * BT + o_hv * BT + j % BT,
                   b_Aqk.to(Aqk.dtype.element_ty), m_hv, MASK_DMA)
            mstore(Akk + i_t * HV * BC + o_hv * BC + j - i_ts,
                   b_Akk.to(Akk.dtype.element_ty), m_hv, MASK_DMA)
    else:
        o_j = tl.arange(0, BJ)
        for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC)), BJ):
            j64 = tl.cast(j, tl.int64)
            rows = j64 + o_j
            m_jhk = (rows[:, None, None] < T) & m_hk[None, :, :]
            p_kj = ((k + j64 * H * K) + o_j[:, None, None] * (H * K)
                    + o_h[None, :, None] * K + o_k[None, None, :])
            p_gj = ((g + j64 * HV * K) + o_j[:, None, None] * (HV * K)
                    + o_hv[None, :, None] * K + o_k[None, None, :])
            b_kj = mload(p_kj, m_jhk, MASK_DMA).to(tl.float32)
            b_gj = mload(p_gj, m_jhk, MASK_DMA).to(tl.float32)
            b_kgj = tl.where(m_k[None, None, :], b_kj * exp2(b_g[None, :, :] - b_gj), 0.0)
            b_Aqk = tl.sum(b_q[None, :, :] * b_kgj, axis=2) * scale
            b_Akk = tl.sum(b_k[None, :, :] * b_kgj, axis=2) * tl.where(rows[:, None] < i_t, 1., 0.)
            p_aq = Aqk + i_t * HV * BT + o_hv[:, None] * BT + j % BT + o_j[None, :]
            p_ak = Akk + i_t * HV * BC + o_hv[:, None] * BC + j - i_ts + o_j[None, :]
            m_out = m_hv[:, None] & (rows[None, :] <= i_t)
            tl.store(p_aq, tl.trans(b_Aqk).to(Aqk.dtype.element_ty), mask=m_out)
            tl.store(p_ak, tl.trans(b_Akk).to(Akk.dtype.element_ty), mask=m_out)


_autotuned_token_parallel = fla_cache_autotune(
    configs=[triton.Config({'BH': bh, 'BJ': bj}) for bh in HEAD_TILES for bj in COLUMN_TILES],
    key=['T', 'N', 'TG_LEN', 'HEAD_LEN', 'UB_CAPACITY', 'H', 'HV', 'K', 'BT', 'BC', 'IS_VARLEN'],
    prune_configs_by={'early_config_prune': prune_head_tiles},
    **autotune_cache_kwargs,
)(chunk_kda_fwd_kernel_intra_token_parallel_npu)


def chunk_kda_fwd_intra_token_parallel_verifier(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
    use_graph: bool = False,
) -> tuple[bool, str | None]:
    chunk_size = 64 if chunk_size is None else int(chunk_size)
    if chunk_size not in (32, 64):
        return False, f'KDA Ascend intra only supports chunk_size in (32, 64), got {chunk_size}'
    return True, None


@register(
    chunk_kda_fwd_intra_token_parallel,
    backend=TritonAscendBackend,
    verifier=chunk_kda_fwd_intra_token_parallel_verifier,
)
@input_guard
def chunk_kda_fwd_intra_token_parallel_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Token-parallel NPU implementation: each token gets its own thread block.
    Supports both fixed-length and variable-length sequences (GVA: HV >= H).

    Writes directly to Aqk and Akk tensors (in-place).
    ``use_graph`` preserves the dispatch signature; token bounds come from ``cu_seqlens[N]`` in both modes.
    """
    B, T, H, K, HV = *q.shape, gk.shape[2]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    # autotuning benchmarks cannot run during graph capture.
    kernel = chunk_kda_fwd_kernel_intra_token_parallel_npu if use_graph else _autotuned_token_parallel
    ub_capacity = 0 if use_graph else get_ub_manager().ub_capacity_bits
    kernel_kwargs = dict(
        q=q, k=k, g=gk, beta=beta, Aqk=Aqk, Akk=Akk, scale=scale,
        cu_seqlens=cu_seqlens, N=N, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC,
        UB_CAPACITY=ub_capacity, IS_VARLEN=cu_seqlens is not None,
    )
    if use_graph:
        kernel_kwargs['BH'] = HEAD_BLOCK_SIZE
    launch_token_parallel_kernel(
        kernel,
        tg_total=B * T,
        hv_total=HV,
        kernel_kwargs=kernel_kwargs,
    )
    return Aqk, Akk
