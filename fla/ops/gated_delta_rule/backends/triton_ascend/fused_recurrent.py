# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size

# b_h [BK,BV] dominates UB usage. BK = next_power_of_2(K) is fixed (NK=1).
# Additional buffers: b_q [BK], b_k [BK], b_v [BV], b_o [BV].
_FUSED_RECURRENT_MEM_MULT = 3.0
_FUSED_RECURRENT_SAFETY_MARGIN = 0.75
_FUSED_RECURRENT_FALLBACK = 8
_FUSED_RECURRENT_MAX_BV = 64
_FUSED_RECURRENT_MIN_BLOCK = 8


def _get_tiles(K: int, V: int) -> tuple[int, int]:
    BK = triton.next_power_of_2(K)
    BV = compute_row_tile_block_size(
        BK, V, _FUSED_RECURRENT_MEM_MULT,
        tiling_row=False,
        safety_margin=_FUSED_RECURRENT_SAFETY_MARGIN,
        fallback=_FUSED_RECURRENT_FALLBACK,
        min_block=_FUSED_RECURRENT_MIN_BLOCK,
        max_block=min(_FUSED_RECURRENT_MAX_BV, triton.next_power_of_2(V)),
    )
    return BK, BV


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_GV": lambda args: args["gv"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_GATE_IN_KERNEL": lambda args: args["A_log"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
    }
)
@triton.jit(do_not_specialize=["scale", "N", "T", "B"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
):
    i_k, i_v = tl.program_id(0), tl.program_id(1)
    i_nh = tl.program_id(2).to(tl.int64)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    if USE_G:
        p_g = g + bos * HV + i_hv

    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k

    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if STATE_V_FIRST:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if STATE_V_FIRST:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V
        if STATE_V_FIRST:
            p_h0 = p_h0 + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = p_h0 + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            if USE_GATE_IN_KERNEL:
                b_A = tl.load(A_log + i_hv).to(tl.float32)
                if HAS_DT_BIAS:
                    b_g = b_g + tl.load(dt_bias + i_hv).to(tl.float32)
                b_g = -exp(b_A) * softplus(b_g)
            b_h *= exp(b_g)
        if USE_GK:
            b_gk = tl.load(p_gk).to(tl.float32)
            if STATE_V_FIRST:
                b_h *= exp(b_gk[None, :])
            else:
                b_h *= exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv).to(tl.float32)
            if STATE_V_FIRST:
                b_h *= exp(b_gv[:, None])
            else:
                b_h *= exp(b_gv[None, :])
        if STATE_V_FIRST:
            b_v -= tl.sum(b_h * b_k[None, :], 1)
        else:
            b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
            if ALLOW_NEG_EIGVAL:
                b_beta = b_beta * 2
        b_v *= b_beta
        if STATE_V_FIRST:
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_h += b_k[:, None] * b_v[None, :]
            b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV * K
        if USE_GV:
            p_gv += HV * V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@input_guard
def fused_recurrent_gated_delta_rule_fwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = _get_tiles(K, V)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        if state_v_first:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None

    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        STATE_V_FIRST=state_v_first,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        ALLOW_NEG_EIGVAL=allow_neg_eigval,
    )
    o = o.squeeze(0)
    return o, final_state
