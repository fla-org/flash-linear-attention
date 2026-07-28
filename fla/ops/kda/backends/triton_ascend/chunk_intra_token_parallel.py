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

from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    max_grid_axis_chunks,
)

_BH = 4
_NUM_WARPS = 2


def _launch_token_parallel_kernel(
    kernel,
    *,
    tg_total: int,
    hv_total: int,
    kernel_kwargs: dict,
) -> None:
    hg_total = triton.cdiv(hv_total, _BH)
    max_tg = max_grid_axis_chunks(tg_total, hg_total, max_grid=ASCEND_MAX_GRID_DIM)
    for tg_off in range(0, tg_total, max_tg):
        tg_len = min(max_tg, tg_total - tg_off)
        kernel_kwargs['TG_OFFSET'] = tg_off
        max_hg = max_grid_axis_chunks(hg_total, tg_len, max_grid=ASCEND_MAX_GRID_DIM)
        for hg_off in range(0, hg_total, max_hg):
            hg_len = min(max_hg, hg_total - hg_off)
            kernel_kwargs['HG_OFFSET'] = hg_off
            kernel[(tg_len, hg_len)](num_warps=_NUM_WARPS, **kernel_kwargs)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'N'])
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
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TG_OFFSET: tl.constexpr,
    HG_OFFSET: tl.constexpr,
):
    i_tg = tl.program_id(0) + TG_OFFSET
    i_hg = tl.program_id(1) + HG_OFFSET

    if IS_VARLEN:
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

        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

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
    o_hv = i_hg * BH + tl.arange(0, BH)
    o_h = o_hv // G
    o_k = tl.arange(0, BK)
    m_hv = o_hv < HV
    m_k = o_k < K
    m_hk = m_hv[:, None] & m_k[None, :]

    p_qk = o_h[:, None] * K + o_k[None, :]
    b_q = tl.load(q + i_t * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)
    b_k = tl.load(k + i_t * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_t * HV * K, (HV, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta + i_t * HV, (HV,), (1,), (i_hg * BH,), (BH,), (0,))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_k = b_k * tl.load(p_beta, boundary_check=(0,)).to(tl.float32)[:, None]

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        b_kj = tl.load(k + j * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)
        p_gj = tl.make_block_ptr(g + j * HV * K, (HV, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

        tl.store(Aqk + i_t * HV * BT + o_hv * BT + j % BT, b_Aqk.to(Aqk.dtype.element_ty), mask=m_hv)
        tl.store(Akk + i_t * HV * BC + o_hv * BC + j - i_ts, b_Akk.to(Akk.dtype.element_ty), mask=m_hv)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Token-parallel NPU implementation: each token gets its own thread block.
    Supports both fixed-length and variable-length sequences (GVA: HV >= H).

    Writes directly to Aqk and Akk tensors (in-place).
    """
    B, T, H, K, HV = *q.shape, gk.shape[2]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    _launch_token_parallel_kernel(
        chunk_kda_fwd_kernel_intra_token_parallel_npu,
        tg_total=B * T,
        hv_total=HV,
        kernel_kwargs=dict(
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            N=N,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BH=_BH,
            TG_OFFSET=0,
            HG_OFFSET=0,
        ),
    )
    return Aqk, Akk
