# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright (c) 2026, Pieter-Jan Hoedt

import torch
import torch.nn.functional as F
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

    i_b = tl.program_id(axis=0).to(tl.int64)
    i_h = tl.program_id(axis=1).to(tl.int64)

    if IS_VARLEN:
        o_seqlens = tl.load(cu_seqlens + i_b + tl.arange(0, 2)).to(tl.int64)
        i_bt, end_bt = tl.split(o_seqlens)
        T = end_bt - i_bt
    else:
        i_bt = i_b * T

    o_1 = tl.arange(0, 1).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    p_q = q + (i_bt * H + i_h) * K + o_k[None, :]
    p_k = k + (i_bt * H + i_h) * K + o_k[None, :]
    p_v = v + (i_bt * H + i_h) * V + o_v[None, :]
    p_log_i = log_i + i_bt * H + i_h + o_1
    p_log_f = log_f + i_bt * H + i_h + o_1
    p_c = c0 + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
    p_m = m0 + i_b * H + i_h

    p_h = h + (i_bt * H + i_h) * V + o_v[None, :]
    p_ms = ms + i_bt * H + i_h + o_1

    if STORE_FINAL_STATE:
        p_c_out = c_out + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        p_m_out = m_out + i_b * H + i_h

    if MAX_NORMALISATION is not None:
        p_n = n0 + (i_b * H + i_h) * K + o_k
        p_z = z + i_bt * H + i_h + o_1
        if STORE_FINAL_STATE:
            p_n_out = n_out + (i_b * H + i_h) * K + o_k

    b_c = tl.load(p_c, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)
    b_m = tl.load(p_m)
    if MAX_NORMALISATION is not None:
        b_n = tl.load(p_n, mask=(o_k < K), other=0)
    for i_t in range(0, T):
        tl.store(p_ms, b_m[None].to(ms.dtype.element_ty))
        b_q = tl.load(p_q, mask=(o_k[None, :] < K), other=0).ravel()
        b_k = qk_scale * tl.load(p_k, mask=(o_k[None, :] < K), other=0).ravel()
        b_v = tl.load(p_v, mask=(o_v[None, :] < V), other=0).ravel()
        b_log_i = tl.load(p_log_i).reshape()
        b_log_f = tl.load(p_log_f).reshape()

        b_log_f += b_m
        b_m = tl.maximum(b_log_f, b_log_i)
        f_gate = tl.exp(b_log_f - b_m)
        i_gate = tl.exp(b_log_i - b_m)
        b_c = f_gate * b_c + i_gate * b_k[:, None] * b_v[None, :]
        b_h = tl.sum(b_q[:, None] * b_c, axis=0)
        if MAX_NORMALISATION is not None:
            b_n = f_gate * b_n + i_gate * b_k
            b_z = tl.sum(b_q * b_n)
            if MAX_NORMALISATION:
                _z = tl.maximum(tl.abs(b_z), tl.exp(-b_m) + eps)
            else:
                _z = b_z + eps

            b_h = b_h / _z

            tl.store(p_z, b_z[None].to(z.dtype.element_ty))
            p_z += H

        tl.store(p_h, b_h[None, :].to(h.dtype.element_ty), mask=(o_v[None, :] < V))

        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_log_i += H
        p_log_f += H
        p_h += H * V
        p_ms += H

    if STORE_FINAL_STATE:
        tl.store(p_c_out, b_c, mask=(o_k[:, None] < K) & (o_v[None, :] < V))
        tl.store(p_m_out, b_m)
        if MAX_NORMALISATION is not None:
            tl.store(p_n_out, b_n, mask=(o_k < K))


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

    i_b = tl.program_id(axis=0).to(tl.int64)
    i_h = tl.program_id(axis=1).to(tl.int64)

    if IS_VARLEN:
        o_seqlens = tl.load(cu_seqlens + i_b + tl.arange(0, 2)).to(tl.int64)
        i_bt, end_bt = tl.split(o_seqlens)
        T = end_bt - i_bt
    else:
        i_bt = i_b * T

    o_1 = tl.arange(0, 1).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    p_dh = dh + (i_bt * H + i_h) * V + o_v[None, :]
    p_q = q + (i_bt * H + i_h) * K + (T - 1 + o_1)[:, None] * (H * K) + o_k[None, :]
    p_k = k + (i_bt * H + i_h) * K + o_k[None, :]
    p_v = v + (i_bt * H + i_h) * V + o_v[None, :]
    p_log_i = log_i + i_bt * H + i_h + o_1
    p_log_f = log_f + (i_bt * H + i_h) + o_1
    p_c0 = c0 + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
    p_m0 = m0 + i_b * H + i_h
    p_ms = ms + (i_bt * H + i_h) + (T - 1 + o_1) * H

    p_dq = dq + (i_bt * H + i_h) * K + o_k[None, :]
    p_dk = dk + (i_bt * H + i_h) * K + (T - 1 + o_1)[:, None] * (H * K) + o_k[None, :]
    p_dv = dv + (i_bt * H + i_h) * V + (T - 1 + o_1)[:, None] * (H * V) + o_v[None, :]
    p_dlog_i = dlog_i + i_bt * H + i_h + (T - 1 + o_1) * H
    p_dlog_f = dlog_f + (i_bt * H + i_h) + (T - 1 + o_1) * H

    if MAX_NORMALISATION is not None:
        p_n0 = n0 + (i_b * H + i_h) * K + o_k
        p_h = h + (i_bt * H + i_h) * V + o_v[None, :]
        p_z = z + i_bt * H + i_h + o_1

    b_c = tl.load(p_c0, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)
    b_m = tl.load(p_m0)
    if MAX_NORMALISATION is not None:
        b_n = tl.load(p_n0, mask=(o_k < K), other=0)
    for i_t in range(0, T):
        b_k = qk_scale * tl.load(p_k, mask=(o_k[None, :] < K), other=0).ravel()
        b_v = tl.load(p_v, mask=(o_v[None, :] < V), other=0).ravel()
        b_log_i = tl.load(p_log_i).reshape()
        b_log_f = tl.load(p_log_f).reshape()

        b_log_f += b_m
        b_m = tl.maximum(b_log_i, b_log_f)
        i_gate = tl.exp(b_log_i - b_m)
        f_gate = tl.exp(b_log_f - b_m)
        b_c = f_gate * b_c + i_gate * b_k[:, None] * b_v[None, :]

        b_dh = tl.load(p_dh, mask=(o_v[None, :] < V), other=0).ravel()
        if MAX_NORMALISATION is not None:
            b_h = tl.load(p_h, mask=(o_v[None, :] < V), other=0).ravel()
            b_n = f_gate * b_n + i_gate * b_k
            b_z = tl.load(p_z).reshape()
            if MAX_NORMALISATION:
                _z_mask = tl.abs(b_z) > tl.exp(-b_m) + eps
                _z = tl.where(_z_mask, tl.abs(b_z), tl.exp(-b_m) + eps)
                z_neg_sign = tl.where(b_z < 0, 1, -1)
                ds = tl.cast(b_dh / _z, dtype=tl.float32)
                dz = _z_mask * z_neg_sign * tl.sum(ds * b_h)
            else:
                _z = b_z + eps
                ds = tl.cast(b_dh / _z, dtype=tl.float32)
                dz = -tl.sum(ds * b_h)

            b_dq = b_n * dz

            p_h += H * V
            p_z += H
        else:
            b_dq = tl.zeros([BK], dtype=tl.float32)
            ds = b_dh

        b_dq += tl.sum(b_c * ds[None, :], axis=1)
        tl.store(p_dq, b_dq[None, :].to(dq.dtype.element_ty), mask=(o_k[None, :] < K))

        p_k += H * K
        p_v += H * V
        p_log_i += H
        p_log_f += H
        p_dh += H * V
        p_dq += H * K

    p_k -= H * K
    p_v -= H * V
    p_log_i -= H
    p_log_f -= H
    p_dh -= H * V
    p_dq -= H * K
    if MAX_NORMALISATION is not None:
        p_h -= H * V
        p_z -= H

    if WITH_FINAL_STATE:
        p_dc = dc + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        p_dc_out = dc_out + (i_b * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]

        b_dc = tl.load(p_dc, mask=(o_k[:, None] < K) & (o_v[None, :] < V), other=0)
        b_df = tl.sum(b_dc * b_c)
        if MAX_NORMALISATION is not None:
            p_dn = dn + (i_b * H + i_h) * K + o_k
            p_dn_out = dn_out + (i_b * H + i_h) * K + o_k

            b_dn = tl.load(p_dn, mask=(o_k < K), other=0)
            b_df += tl.sum(b_dn * b_n)
    else:
        b_df = 0.
        b_dc = tl.zeros([BK, BV], dtype=tl.float32)
        if MAX_NORMALISATION is not None:
            b_dn = tl.zeros([BK], dtype=tl.float32)
    b_m_prev = b_m
    for i_t in range(T - 1, -1, -1):
        b_dh = tl.load(p_dh, mask=(o_v[None, :] < V), other=0).ravel()
        b_dq = tl.load(p_dq, mask=(o_k[None, :] < K), other=0)
        b_q = tl.load(p_q, mask=(o_k[None, :] < K), other=0).ravel()
        b_k = qk_scale * tl.load(p_k, mask=(o_k[None, :] < K), other=0).ravel()
        b_v = tl.load(p_v, mask=(o_v[None, :] < V), other=0).ravel()
        b_log_i = tl.load(p_log_i).reshape()
        b_log_f = tl.load(p_log_f).reshape()

        b_m_prev, b_m = tl.load(p_ms).reshape().to(tl.float32), b_m_prev
        i_gate = tl.exp(b_log_i - b_m)
        f_gate = tl.exp(b_log_f + b_m_prev - b_m)
        if MAX_NORMALISATION is not None:
            b_h = tl.load(p_h, mask=(o_v[None, :] < V), other=0)
            b_z = tl.load(p_z)
            if MAX_NORMALISATION:
                _z_mask = tl.abs(b_z) > tl.exp(-b_m) + eps
                _z = tl.where(_z_mask, tl.abs(b_z), tl.exp(-b_m) + eps)
                z_neg_sign = tl.where(b_z < 0, 1, -1)
                ds = tl.cast(b_dh / _z, dtype=tl.float32)
                dz = _z_mask * z_neg_sign * tl.sum(ds * b_h, axis=1)
            else:
                _z = b_z + eps
                ds = tl.cast(b_dh / _z, dtype=tl.float32)
                dz = -tl.sum(ds * b_h, axis=1)

            b_dn += dz * b_q
            b_dk = i_gate * b_dn
            b_dn *= f_gate

            p_h -= H * V
            p_z -= H
        else:
            ds = b_dh
            b_dk = tl.zeros([BK], dtype=tl.float32)

        b_dc += b_q[:, None] * ds[None, :]
        b_dk += i_gate * tl.sum(b_dc * b_v[None, :], axis=1)
        b_dv = i_gate * tl.sum(b_k[:, None] * b_dc, axis=0)
        b_dc *= f_gate
        b_di = tl.sum(b_dk * b_k)
        # cancellation in this difference requires the FP32 query gradient
        b_df += tl.sum(b_dq * b_q) - b_di
        b_dk *= qk_scale
        tl.store(p_dk, b_dk[None, :].to(dk.dtype.element_ty), mask=(o_k[None, :] < K))
        tl.store(p_dv, b_dv[None, :].to(dv.dtype.element_ty), mask=(o_v[None, :] < V))
        tl.store(p_dlog_i, b_di[None].to(dlog_i.dtype.element_ty))
        tl.store(p_dlog_f, b_df[None].to(dlog_f.dtype.element_ty))

        p_dh -= H * V
        p_dq -= H * K
        p_q -= H * K
        p_k -= H * K
        p_v -= H * V
        p_log_i -= H
        p_log_f -= H
        p_ms -= H
        p_dk -= H * K
        p_dv -= H * V
        p_dlog_i -= H
        p_dlog_f -= H

    if WITH_FINAL_STATE:
        tl.store(p_dc_out, b_dc.to(dc_out.dtype.element_ty), mask=(o_k[:, None] < K) & (o_v[None, :] < V))
        if MAX_NORMALISATION is not None:
            tl.store(p_dn_out, b_dn.to(dn_out.dtype.element_ty), mask=(o_k < K))


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
        log_f=F.logsigmoid(f),
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
        log_f=F.logsigmoid(f),
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
    r"""Compute mLSTM with a fused recurrent scan.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
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
            Gradients are supported for `c` and `n`, but not the stabilizer `m`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.
        max_normalisation (bool | None, Optional):
            Whether to use the maximum-based normalisation of the original paper
            or division by the signed normaliser plus `eps` when `False`.
            If set to `None`, no normalisation will be computed in the kernel
            and the normaliser state will not be updated.
            Default: `True`.

    Returns:
        h (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None):
            Final state in FP32 with same shapes as `initial_state` if `output_final_state=True` else `None`.
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
