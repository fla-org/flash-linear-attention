# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.precond_gated_delta_rule.fused_recurrent import fused_recurrent_precond_gated_delta_rule_fwd
from fla.ops.utils.op import exp
from fla.utils import ascend_compile_kwargs


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_INITIAL_ATK': lambda args: args['a0'] is not None,
    'STORE_FINAL_ATK': lambda args: args['at'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_precond_gated_delta_rule_fwd_kernel_npu(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    g_atk,
    beta_atk,
    log_atk_scale,
    o,
    h0,
    ht,
    a0,
    at,
    cu_seqlens,
    scale,
    x,
    eps,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_INITIAL_ATK: tl.constexpr,
    STORE_FINAL_ATK: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    # ATK pointers (per key head: ATK preconditions k, so g_atk, beta_atk and
    # log_atk_scale carry H heads; under GVA every value head in a group shares them)
    p_g_atk = g_atk + bos * H + i_h
    p_beta_atk = beta_atk + bos * H + i_h

    scale_val = tl.load(log_atk_scale + i_h).to(tl.float32)
    logx = tl.log(tl.cast(x, tl.float32))

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if TRANSPOSE_STATE:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if TRANSPOSE_STATE:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0 = h0 + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # ATK state (per key head: [N, H, K])
    b_a = tl.zeros([BK], dtype=tl.float32)
    if USE_INITIAL_ATK:
        p_a0 = a0 + (i_n * H + i_h) * K + o_k
        b_a += tl.load(p_a0, mask=mask_k, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        # ATK: A_t = exp(g_atk) * A_{t-1} + beta_atk * k^2
        b_g_atk = tl.load(p_g_atk).to(tl.float32)
        b_beta_atk = tl.load(p_beta_atk).to(tl.float32)
        b_a = exp(b_g_atk) * b_a + b_beta_atk * (b_k * b_k)

        # Symmetric fast squash preconditioner
        b_ell = tl.log(b_a + eps)
        b_r = b_ell - scale_val
        b_s = b_r / (1.0 + tl.abs(b_r))
        b_M = tl.exp(-logx * b_s)
        b_k_precond = b_k * b_M

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)

        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h *= exp(b_gk[None, :])
            else:
                b_h *= exp(b_gk[:, None])

        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h *= exp(b_gv[:, None])
            else:
                b_h *= exp(b_gv[None, :])

        # Delta rule update using k_precond for the write side
        if TRANSPOSE_STATE:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
            b_h += b_v[:, None] * b_k_precond[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k_precond[:, None] * b_v
            b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_v += HV*V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV*K
        if USE_GV:
            p_gv += HV*V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV*V
        p_g_atk += H
        p_beta_atk += H

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = ht + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

    if STORE_FINAL_ATK:
        # The ATK state is shared by all value heads in a GVA group; only the
        # first program of each (i_v, group) stores it to avoid duplicate writes.
        if (i_v == 0) and (i_hv % (HV // H) == 0):
            p_at = at + (i_n * H + i_h) * K + o_k
            tl.store(p_at, b_a.to(p_at.dtype.element_ty), mask=mask_k)


def fused_recurrent_precond_gated_delta_rule_fwd_verifier(
    q,
    k,
    v,
    g=None,
    gk=None,
    gv=None,
    beta=None,
    g_atk=None,
    beta_atk=None,
    **kwargs,
) -> tuple[bool, str | None]:
    from fla.utils import IS_NPU
    if not IS_NPU:
        return False, "not running on NPU"
    for tensor in (q, k, v, g, gk, gv, beta, g_atk, beta_atk):
        if isinstance(tensor, torch.Tensor) and tensor.device.type != "npu":
            return False, "input device is not NPU"
    return True, None


@register(
    fused_recurrent_precond_gated_delta_rule_fwd,
    backend=TritonAscendBackend,
    verifier=fused_recurrent_precond_gated_delta_rule_fwd_verifier,
)
def fused_recurrent_precond_gated_delta_rule_fwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    g_atk: torch.Tensor = None,
    beta_atk: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_A_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    if log_atk_scale is None:
        log_atk_scale = torch.full((H,), -0.2, device=k.device, dtype=torch.float32)

    o = torch.empty_like(v)
    if output_final_state:
        if transpose_state_layout:
            final_state = q.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_zeros(N, HV, K, V, dtype=torch.float32)
        final_A_state = q.new_zeros(N, H, K, dtype=torch.float32)
    else:
        final_state = None
        final_A_state = None

    grid = (NV, N * HV)
    fused_recurrent_precond_gated_delta_rule_fwd_kernel_npu[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        g_atk=g_atk,
        beta_atk=beta_atk,
        log_atk_scale=log_atk_scale,
        o=o,
        h0=initial_state,
        ht=final_state,
        a0=initial_A_state,
        at=final_A_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        x=x,
        eps=eps,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        TRANSPOSE_STATE=transpose_state_layout,
        **ascend_compile_kwargs(),
    )
    return o, final_state, final_A_state
