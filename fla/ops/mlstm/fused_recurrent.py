# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright (c) 2026, Pieter-Jan Hoedt

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8] for num_stages in [1, 2, 3]
    ],
    key=['BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_mlstm_fwd_kernel(
    q,
    k,
    v,
    log_i,
    log_f,
    c0,
    n0,
    m0,
    h,
    z,
    ms,
    c_out,
    n_out,
    m_out,
    cu_seqlens,
    qk_scale: float,
    eps: float,
    T: int,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    MAX_NORMALISATION: tl.constexpr,
):
    tl.static_assert(BK >= K, "not implemented yet")
    tl.static_assert(BV >= V, "not implemented yet")

    pid_b = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1).to(tl.int64)

    if IS_VARLEN:
        offsets_t = tl.load(cu_seqlens + pid_b + tl.arange(0, 2)).to(tl.int64)
        pid_bt, end_bt = tl.split(offsets_t)
        T = end_bt - pid_bt
    else:
        pid_bt = pid_b * T

    # input pointers
    o_1 = tl.arange(0, 1).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    q_ptr = q + (pid_bt * H + pid_h) * K + o_k[None, :]
    k_ptr = k + (pid_bt * H + pid_h) * K + o_k[None, :]
    v_ptr = v + (pid_bt * H + pid_h) * V + o_v[None, :]
    log_i_ptr = log_i + pid_bt * H + pid_h + o_1
    log_f_ptr = log_f + pid_bt * H + pid_h + o_1
    c_ptr = c0 + (pid_b * H + pid_h) * K * V + o_k[:, None] * V + o_v[None, :]
    m_ptr = m0 + pid_b * H + pid_h

    # output pointer
    h_ptr = h + (pid_bt * H + pid_h) * V + o_v[None, :]
    ms_ptr = ms + pid_bt * H + pid_h + o_1

    if STORE_FINAL_STATE:
        c_out_ptr = c_out + (pid_b * H + pid_h) * K * V + o_k[:, None] * V + o_v[None, :]
        m_out_ptr = m_out + pid_b * H + pid_h

    if MAX_NORMALISATION is not None:
        n_ptr = n0 + (pid_b * H + pid_h) * K + o_k
        z_ptr = z + pid_bt * H + pid_h + o_1
        if STORE_FINAL_STATE:
            n_out_ptr = n_out + (pid_b * H + pid_h) * K + o_k

    c_vals = tl.load(c_ptr, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)
    m_val = tl.load(m_ptr)
    if MAX_NORMALISATION is not None:
        n_vals = tl.load(n_ptr, mask=(o_k < K), other=0)
    for offset_t in range(0, T):
        tl.store(ms_ptr, m_val[None].to(ms.dtype.element_ty))
        q_vals = tl.load(q_ptr, mask=(o_k[None, :] < K), other=0).ravel()
        k_vals = qk_scale * tl.load(k_ptr, mask=(o_k[None, :] < K), other=0).ravel()
        v_vals = tl.load(v_ptr, mask=(o_v[None, :] < V), other=0).ravel()
        log_i_val = tl.load(log_i_ptr).reshape()
        log_f_val = tl.load(log_f_ptr).reshape()

        log_f_val += m_val
        m_val = tl.maximum(log_f_val, log_i_val)
        f_gate = tl.exp(log_f_val - m_val)
        i_gate = tl.exp(log_i_val - m_val)
        c_vals = f_gate * c_vals + i_gate * k_vals[:, None] * v_vals[None, :]
        h_vals = tl.sum(q_vals[:, None] * c_vals, axis=0)
        if MAX_NORMALISATION is not None:
            n_vals = f_gate * n_vals + i_gate * k_vals
            z_val = tl.sum(q_vals * n_vals)
            if MAX_NORMALISATION:
                _z = tl.maximum(tl.abs(z_val), tl.exp(-m_val) + eps)
            else:
                _z = z_val + eps

            h_vals = h_vals / _z

            tl.store(z_ptr, z_val[None].to(z.dtype.element_ty))
            z_ptr += H

        tl.store(h_ptr, h_vals[None, :].to(h.dtype.element_ty), mask=(o_v[None, :] < V))

        q_ptr += H * K
        k_ptr += H * K
        v_ptr += H * V
        log_i_ptr += H
        log_f_ptr += H
        h_ptr += H * V
        ms_ptr += H

    if STORE_FINAL_STATE:
        tl.store(c_out_ptr, c_vals, mask=(o_k[:, None] < K) & (o_v[None, :] < V))
        tl.store(m_out_ptr, m_val)
        if MAX_NORMALISATION is not None:
            tl.store(n_out_ptr, n_vals, mask=(o_k < K))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8] for num_stages in [1, 2, 3]
    ],
    key=['BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_mlstm_bwd_kernel(
    dh,
    dc,
    dn,
    q,
    k,
    v,
    log_i,
    log_f,
    c0,
    n0,
    m0,
    h,
    z,
    ms,
    dq,
    dk,
    dv,
    dlog_i,
    dlog_f,
    dc_out,
    dn_out,
    cu_seqlens,
    qk_scale: float,
    eps: float,
    T: int,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    WITH_FINAL_STATE: tl.constexpr,
    MAX_NORMALISATION: tl.constexpr,
):
    tl.static_assert(BK >= K, "not implemented yet")
    tl.static_assert(BV >= V, "not implemented yet")

    pid_b = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1).to(tl.int64)

    if IS_VARLEN:
        offsets_t = tl.load(cu_seqlens + pid_b + tl.arange(0, 2)).to(tl.int64)
        pid_bt, end_bt = tl.split(offsets_t)
        T = end_bt - pid_bt
    else:
        pid_bt = pid_b * T

    # input pointers
    o_1 = tl.arange(0, 1).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    dh_ptr = dh + (pid_bt * H + pid_h) * V + o_v[None, :]
    q_ptr = q + (pid_bt * H + pid_h) * K + (T - 1 + o_1)[:, None] * (H * K) + o_k[None, :]
    k_ptr = k + (pid_bt * H + pid_h) * K + o_k[None, :]
    v_ptr = v + (pid_bt * H + pid_h) * V + o_v[None, :]
    log_i_ptr = log_i + pid_bt * H + pid_h + o_1
    log_f_ptr = log_f + (pid_bt * H + pid_h) + o_1
    c0_ptr = c0 + (pid_b * H + pid_h) * K * V + o_k[:, None] * V + o_v[None, :]
    m0_ptr = m0 + pid_b * H + pid_h
    ms_ptr = ms + (pid_bt * H + pid_h) + (T - 1 + o_1) * H

    # output pointers
    dq_ptr = dq + (pid_bt * H + pid_h) * K + o_k[None, :]
    dk_ptr = dk + (pid_bt * H + pid_h) * K + (T - 1 + o_1)[:, None] * (H * K) + o_k[None, :]
    dv_ptr = dv + (pid_bt * H + pid_h) * V + (T - 1 + o_1)[:, None] * (H * V) + o_v[None, :]
    dlog_i_ptr = dlog_i + pid_bt * H + pid_h + (T - 1 + o_1) * H
    dlog_f_ptr = dlog_f + (pid_bt * H + pid_h) + (T - 1 + o_1) * H

    if MAX_NORMALISATION is not None:
        n0_ptr = n0 + (pid_b * H + pid_h) * K + o_k
        h_ptr = h + (pid_bt * H + pid_h) * V + o_v[None, :]
        z_ptr = z + pid_bt * H + pid_h + o_1

    c_vals = tl.load(c0_ptr, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)
    m_val = tl.load(m0_ptr)
    if MAX_NORMALISATION is not None:
        n_vals = tl.load(n0_ptr, mask=(o_k < K), other=0)
    for offset_t in range(0, T):
        k_vals = qk_scale * tl.load(k_ptr, mask=(o_k[None, :] < K), other=0).ravel()
        v_vals = tl.load(v_ptr, mask=(o_v[None, :] < V), other=0).ravel()
        log_i_val = tl.load(log_i_ptr).reshape()
        log_f_val = tl.load(log_f_ptr).reshape()

        log_f_val += m_val
        m_val = tl.maximum(log_i_val, log_f_val)
        i_gate = tl.exp(log_i_val - m_val)
        f_gate = tl.exp(log_f_val - m_val)
        c_vals = f_gate * c_vals + i_gate * k_vals[:, None] * v_vals[None, :]

        dh_vals = tl.load(dh_ptr, mask=(o_v[None, :] < V), other=0).ravel()
        if MAX_NORMALISATION is not None:
            h_vals = tl.load(h_ptr, mask=(o_v[None, :] < V), other=0).ravel()
            n_vals = f_gate * n_vals + i_gate * k_vals
            z_val = tl.load(z_ptr).reshape()
            if MAX_NORMALISATION:
                _z_mask = tl.abs(z_val) > tl.exp(-m_val) + eps
                _z = tl.where(_z_mask, tl.abs(z_val), tl.exp(-m_val) + eps)
                z_neg_sign = tl.where(z_val < 0, 1, -1)
                ds = tl.cast(dh_vals / _z, dtype=tl.float32)
                dz = _z_mask * z_neg_sign * tl.sum(ds * h_vals)
            else:
                _z = z_val + eps
                ds = tl.cast(dh_vals / _z, dtype=tl.float32)
                dz = -tl.sum(ds * h_vals)

            dq_vals = n_vals * dz

            h_ptr += H * V
            z_ptr += H
        else:
            dq_vals = tl.zeros([BK], dtype=tl.float32)
            ds = dh_vals

        dq_vals += tl.sum(c_vals * ds[None, :], axis=1)
        tl.store(dq_ptr, dq_vals[None, :].to(dq.dtype.element_ty), mask=(o_k[None, :] < K))

        k_ptr += H * K
        v_ptr += H * V
        log_i_ptr += H
        log_f_ptr += H
        dh_ptr += H * V
        dq_ptr += H * K

    # prepare pointers for reuse
    k_ptr -= H * K
    v_ptr -= H * V
    log_i_ptr -= H
    log_f_ptr -= H
    dh_ptr -= H * V
    dq_ptr -= H * K
    if MAX_NORMALISATION is not None:
        h_ptr -= H * V
        z_ptr -= H

    if WITH_FINAL_STATE:
        dc_ptr = dc + (pid_b * H + pid_h) * K * V + o_k[:, None] * V + o_v[None, :]
        dc_out_ptr = dc_out + (pid_b * H + pid_h) * K * V + o_k[:, None] * V + o_v[None, :]

        dc_vals = tl.load(dc_ptr, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)
        df_val = tl.sum(dc_vals * c_vals)
        if MAX_NORMALISATION is not None:
            dn_ptr = dn + (pid_b * H + pid_h) * K + o_k
            dn_out_ptr = dn_out + (pid_b * H + pid_h) * K + o_k

            dn_vals = tl.load(dn_ptr, mask=(o_k < K), other=0)
            df_val += tl.sum(dn_vals * n_vals)
    else:
        df_val = 0.
        dc_vals = tl.zeros([BK, BV], dtype=tl.float32)
        if MAX_NORMALISATION is not None:
            dn_vals = tl.zeros([BK], dtype=tl.float32)
    m_prev_val = m_val
    for offset_t in range(T - 1, -1, -1):
        dh_vals = tl.load(dh_ptr, mask=(o_v[None, :] < V), other=0).ravel()
        dq_vals = tl.load(dq_ptr, mask=(o_k[None, :] < K), other=0)
        q_vals = tl.load(q_ptr, mask=(o_k[None, :] < K), other=0).ravel()
        k_vals = qk_scale * tl.load(k_ptr, mask=(o_k[None, :] < K), other=0).ravel()
        v_vals = tl.load(v_ptr, mask=(o_v[None, :] < V), other=0).ravel()
        log_i_val = tl.load(log_i_ptr).reshape()
        log_f_val = tl.load(log_f_ptr).reshape()

        m_prev_val, m_val = tl.load(ms_ptr).reshape().to(tl.float32), m_prev_val
        i_gate = tl.exp(log_i_val - m_val)
        f_gate = tl.exp(log_f_val + m_prev_val - m_val)
        if MAX_NORMALISATION is not None:
            h_vals = tl.load(h_ptr, mask=(o_v[None, :] < V), other=0)
            z_val = tl.load(z_ptr)
            if MAX_NORMALISATION:
                _z_mask = tl.abs(z_val) > tl.exp(-m_val) + eps
                _z = tl.where(_z_mask, tl.abs(z_val), tl.exp(-m_val) + eps)
                z_neg_sign = tl.where(z_val < 0, 1, -1)
                ds = tl.cast(dh_vals / _z, dtype=tl.float32)
                dz = _z_mask * z_neg_sign * tl.sum(ds * h_vals, axis=1)
            else:
                _z = z_val + eps
                ds = tl.cast(dh_vals / _z, dtype=tl.float32)
                dz = -tl.sum(ds * h_vals, axis=1)

            dn_vals += dz * q_vals
            dk_vals = i_gate * dn_vals
            dn_vals *= f_gate

            h_ptr -= H * V
            z_ptr -= H
        else:
            ds = dh_vals
            dk_vals = tl.zeros([BK], dtype=tl.float32)

        dc_vals += q_vals[:, None] * ds[None, :]
        dk_vals += i_gate * tl.sum(dc_vals * v_vals[None, :], axis=1)
        dv_vals = i_gate * tl.sum(k_vals[:, None] * dc_vals, axis=0)
        dc_vals *= f_gate
        di_val = tl.sum(dk_vals * k_vals)
        df_val += tl.sum(dq_vals * q_vals) - di_val  # catastrophic forgetting
        dk_vals *= qk_scale
        tl.store(dk_ptr, dk_vals[None, :].to(dk.dtype.element_ty), mask=(o_k[None, :] < K))
        tl.store(dv_ptr, dv_vals[None, :].to(dv.dtype.element_ty), mask=(o_v[None, :] < V))
        tl.store(dlog_i_ptr, di_val[None].to(dlog_i.dtype.element_ty))
        tl.store(dlog_f_ptr, df_val[None].to(dlog_f.dtype.element_ty))

        dh_ptr -= H * V
        dq_ptr -= H * K
        q_ptr -= H * K
        k_ptr -= H * K
        v_ptr -= H * V
        log_i_ptr -= H
        log_f_ptr -= H
        ms_ptr -= H
        dk_ptr -= H * K
        dv_ptr -= H * V
        dlog_i_ptr -= H
        dlog_f_ptr -= H

    if WITH_FINAL_STATE:
        tl.store(dc_out_ptr, dc_vals.to(dc_out.dtype.element_ty), mask=(o_k[:, None] < K) & (o_v[None, :] < V))
        if MAX_NORMALISATION is not None:
            tl.store(dn_out_ptr, dn_vals.to(dn_out.dtype.element_ty), mask=(o_k < K))


def fused_recurrent_mlstm_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    max_normalisation: bool | None = True,
):
    B, T, H, K, V = *k.shape, v.shape[-1]

    if cu_seqlens is not None:
        B = len(cu_seqlens) - 1

    if initial_state is None:
        state_dtype = torch.float32
        c0 = torch.zeros(B, H, K, V, dtype=state_dtype, device=k.device)
        n0 = None
        m0 = torch.full((B, H), float('-inf'), dtype=state_dtype, device=k.device)
        if max_normalisation is not None:
            n0 = torch.zeros(B, H, K, dtype=state_dtype, device=k.device)
    else:
        c0, n0, m0 = initial_state

    if output_final_state:
        c_out = torch.empty_like(c0)
        n_out = torch.empty_like(n0) if max_normalisation is not None else n0
        m_out = torch.empty_like(m0)
    else:
        c_out, n_out, m_out = None, None, None

    h = torch.empty_like(v)
    z = torch.empty_like(i, dtype=torch.float32)
    ms = torch.empty_like(i, dtype=torch.float32)
    fused_recurrent_mlstm_fwd_kernel[(B, H)](
        q=q,
        k=k,
        v=v,
        log_i=i,
        log_f=torch.nn.functional.logsigmoid(f),
        c0=c0,
        n0=n0,
        m0=m0,
        h=h,
        z=z,
        ms=ms,
        c_out=c_out,
        n_out=n_out,
        m_out=m_out,
        cu_seqlens=cu_seqlens,
        qk_scale=scale,
        eps=eps,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=max(16, triton.next_power_of_2(K)),
        BV=max(16, triton.next_power_of_2(V)),
        IS_VARLEN=cu_seqlens is not None,
        STORE_FINAL_STATE=c_out is not None,
        MAX_NORMALISATION=max_normalisation,
    )

    return h, c_out, n_out, m_out, z, ms


def fused_recurrent_mlstm_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    h: torch.Tensor | None,
    z: torch.Tensor | None,
    ms: torch.Tensor,
    dh: torch.Tensor,
    dc: torch.Tensor,
    dn: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    max_normalisation: bool | None = True,
):
    B, T, H, K, V = *k.shape, v.shape[-1]

    if cu_seqlens is not None:
        B = len(cu_seqlens) - 1

    if initial_state is None:
        state_dtype = torch.float32
        c0 = torch.zeros(B, H, K, V, dtype=state_dtype, device=k.device)
        n0 = None
        m0 = torch.full((B, H), float('-inf'), dtype=state_dtype, device=k.device)
        if max_normalisation is not None:
            n0 = torch.zeros(B, H, K, dtype=state_dtype, device=k.device)
    else:
        c0, n0, m0 = initial_state
        # seed absent final-state gradients with zero to propagate gradients to the initial state
        if dc is None:
            dc = torch.zeros_like(c0)
        if max_normalisation is not None and dn is None:
            dn = torch.zeros_like(n0)

    state_dtype = torch.float32
    dc_out = torch.empty_like(dc, dtype=state_dtype) if dc is not None else None
    if max_normalisation is not None:
        dn_out = torch.empty_like(dn, dtype=state_dtype) if dn is not None else None
    else:
        dn_out = dn

    # the query gradient is reused in the forget-gate gradient before conversion to the input dtype
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dlog_i = torch.empty_like(i)
    dlog_f = torch.empty_like(f)
    fused_recurrent_mlstm_bwd_kernel[(B, H)](
        dh=dh,
        dc=dc,
        dn=dn,
        q=q,
        k=k,
        v=v,
        log_i=i,
        log_f=torch.nn.functional.logsigmoid(f),
        c0=c0,
        n0=n0,
        m0=m0,
        h=h,
        z=z,
        ms=ms,
        dq=dq,
        dk=dk,
        dv=dv,
        dlog_i=dlog_i,
        dlog_f=dlog_f,
        dc_out=dc_out,
        dn_out=dn_out,
        cu_seqlens=cu_seqlens,
        qk_scale=scale,
        eps=eps,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=max(16, triton.next_power_of_2(K)),
        BV=max(16, triton.next_power_of_2(V)),
        IS_VARLEN=cu_seqlens is not None,
        WITH_FINAL_STATE=dc is not None,
        MAX_NORMALISATION=max_normalisation,
    )
    df = torch.sigmoid(-f) * dlog_f

    return dq.to(q.dtype), dk, dv, dlog_i, df, dc_out, dn_out


class FusedRecurrentMLSTMFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        i,
        f,
        initial_c: torch.Tensor | None,
        initial_n: torch.Tensor | None,
        initial_m: torch.Tensor | None,
        scale: float,
        eps: float,
        output_final_state: bool,
        cu_seqlens,
        max_normalisation: bool | None,
    ):
        initial_state = None if initial_c is None else (initial_c, initial_n, initial_m)
        h, c, n, m, z, ms = fused_recurrent_mlstm_fwd(
            q=q,
            k=k,
            v=v,
            i=i,
            f=f,
            scale=scale,
            eps=eps,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            max_normalisation=max_normalisation,
        )
        if m is not None:
            ctx.mark_non_differentiable(m)
        ctx.save_for_backward(q, k, v, i, f, h, z, ms)
        ctx.initial_state = initial_state
        ctx.scale = scale
        ctx.eps = eps
        ctx.cu_seqlens = cu_seqlens
        ctx.max_normalisation = max_normalisation
        return h, c, n, m

    @staticmethod
    @input_guard
    @torch.autograd.function.once_differentiable
    @autocast_custom_bwd
    def backward(ctx, dh, dc, dn, dm):
        q, k, v, i, f, h, z, ms = ctx.saved_tensors
        dq, dk, dv, di, df, dc_out, dn_out = fused_recurrent_mlstm_bwd(
            q=q,
            k=k,
            v=v,
            i=i,
            f=f,
            h=h,
            z=z,
            ms=ms,
            dh=dh,
            dc=dc,
            dn=dn,
            scale=ctx.scale,
            eps=ctx.eps,
            initial_state=ctx.initial_state,
            cu_seqlens=ctx.cu_seqlens,
            max_normalisation=ctx.max_normalisation,
        )

        if ctx.initial_state is None:
            dc_out, dn_out = None, None

        return dq, dk, dv, di, df, dc_out, dn_out, None, None, None, None, None, None


def fused_recurrent_mlstm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor | None = None,
    f: torch.Tensor | None = None,
    scale: float | None = None,
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    max_normalisation: bool | None = True,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        i (torch.Tensor):
            Input gate logits of shape `[B, T, H]`.
        f (torch.Tensor):
            Forget gate logits of shape `[B, T, H]`.
        scale (float, Optional):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        eps (float, Optional):
            Stabiliser for the normalisation denominator. Default: 1e-6.
        initial_state (tuple[torch.Tensor, torch.Tensor | None, torch.Tensor], Optional):
            Initial state `(c0, n0, m0)` for `N` input sequences,
            with shapes `[N, H, K, V]`, `[N, H, K]`, and `[N, H]`, respectively.
            `n0` may be `None` when `max_normalisation` is `None`.
            For equal-length input sequences, `N` equals the batch size `B`.
            State tensors are converted to FP32 for computation. Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final state tuple `(c, n, m)`. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.
        max_normalisation (bool | None, Optional):
            Whether to use the maximum-based normalisation of the original paper
            or a regular FLA normalisation inside the kernel.
            If set to `None`, no normalisation will be computed in the kernel
            and the normaliser state will not be updated.
            Default: `True`.

    Returns:
        h (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None):
            Final state in FP32 with same shapes as `initial_state` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from fla.ops.mlstm import fused_recurrent_mlstm
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> i = torch.randn(B, T, H, device='cuda')
        >>> f = torch.randn(B, T, H, device='cuda')
        >>> c0 = torch.randn(B, H, K, V, device='cuda')
        >>> n0 = torch.randn(B, H, K, device='cuda')
        >>> m0 = torch.randn(B, H, device='cuda')
        >>> h, ct = fused_recurrent_mlstm(
            q, k, v, i, f,
            initial_state=(c0, n0, m0),
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v = map(lambda x: x.flatten(end_dim=1).unsqueeze(0), (q, k, v))
        >>> i, f = map(lambda x: x.flatten(end_dim=1).unsqueeze(0), (i, f))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> h_var, ct_var = fused_recurrent_mlstm(
            q, k, v, i, f,
            initial_state=(c0, n0, m0),
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
        >>> assert h.allclose(h_var.view(h.shape))
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and any(
            s.shape[0] != len(cu_seqlens) - 1 for s in initial_state if s is not None
        ):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {tuple(None if s is None else s.shape[0] for s in initial_state)}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if initial_state is None:
        initial_state = (None, None, None)
    else:
        initial_state = tuple(state.float() if state is not None else None for state in initial_state)
    initial_c, initial_n, initial_m = initial_state
    h, c, n, m = FusedRecurrentMLSTMFunction.apply(
        q, k, v, i, f, initial_c, initial_n, initial_m,
        scale, eps, output_final_state, cu_seqlens, max_normalisation
    )
    return h, (c, n, m) if output_final_state else None
