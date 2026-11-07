# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend fused recurrent forward for the gated Oja rule.

On Ascend A2/A3 the mainline kernel's single ``[BK, V]`` fp32 state tile plus
its per-step temporaries exceeds the UB budget at ``K = V = 128``. The Oja
residual ``k - h @ v`` couples all V columns, so V cannot be split across
programs; instead the state lives as sequential in-kernel V slabs of width
``BV``, and the residual is summed slab by slab.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size

# Peak UB footprint per [BK, BV] fp32 tile on A2/A3, derived not inherited:
# the single-slab mainline form measures ~3.5x its tile (state slab plus
# temporaries); this kernel keeps two slabs live, so one extra state copy
# gives ~4.5.
_RECUR_MEM_MULT = 4.5
_SAFETY_MARGIN = 0.80
_FALLBACK_BV = 32
# blockdim64 house style: slab width caps at 64; the wrapper asserts V <= 128,
# so at most two slabs are ever live.
_MAX_BV = 64


def _get_bv(K: int, V: int) -> int:
    bk = triton.next_power_of_2(K)
    return compute_row_tile_block_size(
        bk,
        V,
        _RECUR_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BV,
        min_block=1,
        max_block=min(_MAX_BV, triton.next_power_of_2(V)),
    )


@triton.heuristics({
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_oja_fwd_kernel_npu(
    q,
    k,
    v,
    gv,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_Q_L2NORM: tl.constexpr,
    USE_K_L2NORM: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T
    o_k = tl.arange(0, BK)
    o_v1 = tl.arange(0, BV)
    o_v2 = BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v1 = v + (bos * HV + i_hv) * V + o_v1
    p_v2 = v + (bos * HV + i_hv) * V + o_v2
    if USE_GV:
        p_gv1 = gv + (bos * HV + i_hv) * V + o_v1
        p_gv2 = gv + (bos * HV + i_hv) * V + o_v2
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta1 = beta + (bos * HV + i_hv) * V + o_v1
        p_beta2 = beta + (bos * HV + i_hv) * V + o_v2

    p_o1 = o + (bos * HV + i_hv) * V + o_v1
    p_o2 = o + (bos * HV + i_hv) * V + o_v2

    mask_k = o_k < K
    mask_v1 = o_v1 < V
    mask_v2 = o_v2 < V

    b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
    if V > BV:
        b_h2 = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V
        b_h1 += tl.load(p_h0 + o_v1[None, :], mask=mask_k[:, None] & mask_v1[None, :], other=0).to(tl.float32)
        if V > BV:
            b_h2 += tl.load(p_h0 + o_v2[None, :], mask=mask_k[:, None] & mask_v2[None, :], other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v1 = tl.load(p_v1, mask=mask_v1, other=0).to(tl.float32)
        if V > BV:
            b_v2 = tl.load(p_v2, mask=mask_v2, other=0).to(tl.float32)
        if USE_Q_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        if USE_K_L2NORM:
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta1 = tl.load(p_beta1, mask=mask_v1, other=0).to(tl.float32)
            if V > BV:
                b_beta2 = tl.load(p_beta2, mask=mask_v2, other=0).to(tl.float32)

        if USE_GV:
            b_gv1 = tl.load(p_gv1, mask=mask_v1, other=0).to(tl.float32)
            b_h1 *= exp(b_gv1[None, :])
            if V > BV:
                b_gv2 = tl.load(p_gv2, mask=mask_v2, other=0).to(tl.float32)
                b_h2 *= exp(b_gv2[None, :])

        # the residual couples every V column; sum it slab by slab
        if IS_BETA_HEADWISE:
            b_k = b_beta * (b_k - tl.sum(b_h1 * b_v1[None, :], 1))
            if V > BV:
                b_k = b_k - b_beta * tl.sum(b_h2 * b_v2[None, :], 1)
            b_h1 += b_k[:, None] * b_v1
            if V > BV:
                b_h2 += b_k[:, None] * b_v2
        else:
            # per-column beta scales the v side: h[k, v] += beta[v] * res[k] * v[v];
            # folded into b_v so the update stays [BK] x [BV] for any slab split
            b_k = b_k - tl.sum(b_h1 * b_v1[None, :], 1)
            if V > BV:
                b_k = b_k - tl.sum(b_h2 * b_v2[None, :], 1)
            b_h1 += b_k[:, None] * (b_beta1 * b_v1)[None, :]
            if V > BV:
                b_h2 += b_k[:, None] * (b_beta2 * b_v2)[None, :]

        b_o1 = tl.sum(b_h1 * b_q[:, None], 0)
        tl.store(p_o1, b_o1.to(p_o1.dtype.element_ty), mask=mask_v1)
        if V > BV:
            b_o2 = tl.sum(b_h2 * b_q[:, None], 0)
            tl.store(p_o2, b_o2.to(p_o2.dtype.element_ty), mask=mask_v2)

        p_q += H*K
        p_k += H*K
        p_v1 += HV*V
        p_v2 += HV*V
        if USE_GV:
            p_gv1 += HV*V
            p_gv2 += HV*V
        if IS_BETA_HEADWISE:
            p_beta += HV
        else:
            p_beta1 += HV*V
            p_beta2 += HV*V
        p_o1 += HV*V
        p_o2 += HV*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V
        tl.store(p_ht + o_v1[None, :], b_h1.to(p_ht.dtype.element_ty), mask=mask_k[:, None] & mask_v1[None, :])
        if V > BV:
            tl.store(p_ht + o_v2[None, :], b_h2.to(p_ht.dtype.element_ty), mask=mask_k[:, None] & mask_v2[None, :])


@input_guard
def fused_recurrent_oja_fwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_q_l2norm: bool = False,
    use_k_l2norm: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    assert V <= 128
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = _get_bv(K, V)
    # the state lives in at most two in-kernel V slabs of width BV
    assert BV * 2 >= V, 'current kernel supports at most two V slabs.'

    o = torch.empty_like(v)
    final_state = q.new_empty(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    grid = (N * HV,)
    fused_recurrent_oja_fwd_kernel_npu[grid](
        q=q,
        k=k,
        v=v,
        gv=gv,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_Q_L2NORM=use_q_l2norm,
        USE_K_L2NORM=use_k_l2norm,
        **ascend_compile_kwargs(),
    )
    return o, final_state
