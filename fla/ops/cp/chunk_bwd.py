# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_delta_h_state import chunk_delta_rule_bwd_dh_state
from fla.ops.cp.chunk_merge import _prune_cp_configs
from fla.ops.cp.context import _check_cp_device
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs, check_shared_mem


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([4, 3, 2, 1] if check_shared_mem('ampere') else [1])
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'USE_G', 'USE_GK', 'USE_BG', 'IS_VARLEN'],
    prune_configs_by={'early_config_prune': _prune_cp_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_cp_bwd_kernel_dh(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Summarize local backward propagation as dh_in = dM @ dh_out + dh_local.

    Each value head writes [dh_local | dM] with shape [K, V+K].
    Both the zero-terminal-gradient response and its boundary-gradient transform scan chunks in reverse order.
    """
    i_col, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    is_dh_part = i_col * BLOCK_SIZE < V

    # for DPLR, use H heads for w; GDN and KDA use HV heads for w
    q += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    dhm += i_h * K * (V + K)
    stride_qk = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_dh_part:
        do += ((bos * HV + i_h) * V).to(tl.int64)
        dv += ((bos * HV + i_h) * V).to(tl.int64)
        i_v = i_col

        b_dh2, b_dh3, b_dh4 = tl.full((), 0, tl.float32), tl.full((), 0, tl.float32), tl.full((), 0, tl.float32)
        # zero terminal gradient isolates contributions from this rank's do and dv
        b_dh1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        o_vb = i_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m_vb = o_vb < V
        o_k1 = tl.arange(0, 64)
        m_k1 = o_k1 < K
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        o_k3 = 128 + o_k1
        m_k3 = o_k3 < K
        o_k4 = 192 + o_k1
        m_k4 = o_k4 < K

        for i_t in range(NT - 1, -1, -1):
            last_idx = min((i_t + 1) * BT, T) - 1
            o_t = i_t * BT + tl.arange(0, BT)

            # reuse the common recurrence with zero terminal gradient to form the local affine offset
            b_dh1, b_dh2, b_dh3, b_dh4 = chunk_delta_rule_bwd_dh_state(
                q=q,
                k=k,
                w=w,
                g=g,
                gk=gk,
                do=do,
                dv=dv,
                dv2=None,
                b_dh=(b_dh1, b_dh2, b_dh3, b_dh4),
                o_t=o_t,
                o_v=o_vb,
                last_idx=last_idx,
                T=T,
                scale=scale,
                H=H,
                HV=HV,
                K=K,
                V=V,
                USE_G=USE_G,
                USE_GK=USE_GK,
                USE_BG=USE_BG,
                SAVE_NEW_VALUE=False,
                STATE_V_FIRST=False,
            )

        p_dh1 = dhm + o_k1[:, None] * (V + K) + o_vb[None, :]
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_k1[:, None] & m_vb[None, :])
        if K > 64:
            p_dh2 = dhm + o_k2[:, None] * (V + K) + o_vb[None, :]
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :])
        if K > 128:
            p_dh3 = dhm + o_k3[:, None] * (V + K) + o_vb[None, :]
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :])
        if K > 192:
            p_dh4 = dhm + o_k4[:, None] * (V + K) + o_vb[None, :]
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :])
    else:
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for _i_t in range(NT):
            i_t = NT - 1 - _i_t
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T

            p_k = k + o_t[:, None] * stride_qk + row[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
            p_w = w + o_t[:, None] * stride_w + row[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
                b_gk_last = exp2(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            # backward transposes each chunk's state transform; DPLR passes its gated bg through k
            b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))
            if USE_BG:
                # DPLR: dM = (diag + w^T @ bg) @ dM
                b_m_i = b_diag + b_kw
            else:
                # GDN/KDA: dM = (diag - w^T @ k) @ dM
                b_m_i = b_diag - b_kw
            # keep the matrix chain in fp32 to avoid repeated bf16 rounding
            b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32), input_precision="tf32x3")

        p_m = dhm + V + row[:, None] * (V + K) + col[None, :]
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=(row < K)[:, None] & (col < K)[None, :])


def chunk_cp_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    dhm: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> None:
    """Write the backward affine summary of one local sequence into dhm."""
    _check_cp_device(k.device)
    _, T, H, K = q.shape
    HV, V = do.shape[2:]
    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64
    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
    chunk_cp_bwd_kernel_dh[grid](
        q=q,
        k=k if bg is None else bg,
        w=w,
        g=g,
        gk=gk,
        do=do,
        dhm=dhm,
        dv=dv,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BK1=BK,
        USE_BG=bg is not None,
    )
