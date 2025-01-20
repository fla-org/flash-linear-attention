# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.common.fused_recurrent import (fused_recurrent_bwd_kernel,
                                            fused_recurrent_fwd_kernel)
from fla.ops.utils import chunk_global_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.jit
def fused_recurrent_gsa_inference_kernel(
    q,
    k,
    v,
    s,
    g,
    o,
    hk0,
    hv0,
    hkt,
    hvt,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG

    b_s = tl.load(s + i_bg * M + tl.arange(0, M)).to(tl.float32)
    b_g = tl.load(g + i_bg * M + tl.arange(0, M)).to(tl.float32)
    b_g = tl.exp(b_g)

    b_ok = tl.zeros([M], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)

        p_hk0 = hk0 + i_bg * K * M + (o_k[None, :]) * M + tl.arange(0, M)[:, None]
        # [BK,]
        mask_k = o_k < K
        # [M, BK]
        mask_hk = (tl.arange(0, M) < M)[:, None] & mask_k[None, :]
        # [M, BK]
        b_hk = tl.load(p_hk0, mask=mask_hk, other=0.).to(tl.float32)
        # [BK,]
        b_q = tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.).to(tl.float32) * scale
        b_k = tl.load(k + i_bg * K + o_k, mask=mask_k, other=0.).to(tl.float32)
        b_hk = b_hk * b_g[:, None] + b_k[None, :] * b_s[:, None]
        b_ok += tl.sum(b_hk * b_q[None, :], axis=1)

        if i_bh % NG == 0:
            p_hkt = hkt + i_bg * K * M + o_k[None, :] * M + tl.arange(0, M)[:, None]
            tl.store(p_hkt, b_hk.to(p_hkt.dtype.element_ty), mask=mask_hk)

    b_qv = tl.softmax(b_ok)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)

        p_hv0 = hv0 + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None]
        # [BV,]
        mask_v = o_v < V
        # [BV, M]
        mask_hv = mask_v[:, None] & (tl.arange(0, M) < M)[None, :]
        # [BV, M]
        b_hv = tl.load(p_hv0, mask=mask_hv, other=0).to(tl.float32)
        # [BV,]
        b_v = tl.load(v + i_bg * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_hv = b_hv * b_g[None, :] + b_s[None, :] * b_v[:, None]
        b_ov = tl.sum(b_hv * b_qv[None, :], axis=1)

        tl.store(o + i_bh * V + o_v, b_ov.to(o.dtype.element_ty), mask=mask_v)

        if i_bh % NG == 0:
            p_hvt = hvt + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None]
            tl.store(p_hvt, b_hv.to(p_hvt.dtype.element_ty), mask=mask_hv)


def fused_recurrent_gsa_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: float = 1.,
    head_first: bool = True
) -> torch.Tensor:
    if head_first:
        B, H, T, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    HQ = q.shape[1] if head_first else q.shape[2]
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NG = HQ // H

    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    hkt, hvt = None, None
    if output_final_state:
        if NG == 1:
            hkt, hvt = hk0, hv0
        else:
            hkt, hvt = q.new_empty(B, H, K, M, dtype=torch.float), q.new_empty(B, H, M, V, dtype=torch.float)

    o = v.new_empty(B, HQ, T, V) if head_first else v.new_empty(B, T, HQ, V)
    grid = (B * HQ,)
    fused_recurrent_gsa_inference_kernel[grid](
        q,
        k,
        v,
        s,
        g,
        o,
        hk0,
        hv0,
        hkt,
        hvt,
        scale=scale,
        K=K,
        V=V,
        M=M,
        BK=BK,
        BV=BV,
        NG=NG
    )
    return o, (hkt, hvt)


def fused_recurrent_gsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: float = 1.,
    reverse: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
    if head_first:
        B, H, T, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    HQ = q.shape[1] if head_first else q.shape[2]
    if HQ != H:
        raise ValueError("GQA not supported yet.")

    BK, BV, BM = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64), min(M, 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)

    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    hkt, hvt = None, None
    if output_final_state:
        hkt, hvt = q.new_empty(N, H, K, M, dtype=torch.float), q.new_empty(N, H, M, V, dtype=torch.float)

    ok = q.new_empty(NK, *s.shape, dtype=torch.float)
    gk, gv = None, g
    grid = (NM, NK, N * H)
    fused_recurrent_fwd_kernel[grid](
        q=q,
        k=k,
        v=s,
        g=None,
        gk=gk,
        gv=gv,
        o=ok,
        h0=hk0,
        ht=hkt,
        offsets=offsets,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=M,
        BK=BK,
        BV=BM,
        USE_G=False,
        USE_GK=False,
        USE_GV=True,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    ok = ok.sum(0)

    qv = ok.softmax(-1, dtype=torch.float)
    ov = q.new_empty(NM, *v.shape, dtype=torch.float)
    gk, gv = g, None
    grid = (NV, NM, N * H)
    fused_recurrent_fwd_kernel[grid](
        q=qv,
        k=s,
        v=v,
        g=None,
        gk=gk,
        gv=gv,
        o=ov,
        h0=hv0,
        ht=hvt,
        offsets=offsets,
        scale=1.,
        B=B,
        T=T,
        H=H,
        K=M,
        V=V,
        BK=BM,
        BV=BV,
        USE_G=False,
        USE_GK=True,
        USE_GV=False,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    ov = ov.sum(0)
    return ok, hkt, qv, ov, hvt


def fused_recurrent_gsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    qv: torch.Tensor,
    hk0: Optional[torch.Tensor] = None,
    hv0: Optional[torch.Tensor] = None,
    ok: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    dhkt: Optional[torch.Tensor] = None,
    dhvt: Optional[torch.Tensor] = None,
    scale: float = 1.,
    reverse: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor]:
    if head_first:
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
    N = B if offsets is None else len(offsets) - 1

    BK, BV, BM = min(K, 64), min(V, 64), min(M, 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)

    if head_first:
        dqv = q.new_empty(NV, B, H, T, M, dtype=torch.float)
        dsv = q.new_empty(NV, B, H, T, M, dtype=torch.float)
        dv = q.new_empty(NM, B, H, T, V, dtype=torch.float)
    else:
        dqv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
        dsv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
        dv = q.new_empty(NM, B, T, H, V, dtype=torch.float)
    dhk0 = torch.empty_like(hk0)if hk0 is not None else None
    dhv0 = torch.empty_like(hv0)if hv0 is not None else None

    gk, gv = g, None
    grid = (NV, NM, N * H)
    fused_recurrent_bwd_kernel[grid](
        q=qv,
        k=s,
        v=v,
        g=None,
        gk=gk,
        gv=gv,
        h0=hv0,
        do=do,
        dq=dqv,
        dk=dsv,
        dv=dv,
        dht=dhvt,
        dh0=dhv0,
        offsets=offsets,
        scale=1.,
        B=B,
        T=T,
        H=H,
        K=M,
        V=V,
        BK=BM,
        BV=BV,
        USE_G=False,
        USE_GK=True,
        USE_GV=False,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    dqv = dqv.sum(0)
    dsv = dsv.sum(0)
    dv = dv.sum(0)
    dgk = chunk_global_cumsum(dqv * qv.float() - dsv * s.float(),
                              reverse=not reverse,
                              offsets=offsets,
                              head_first=head_first)

    dok = qv * (dqv - (qv * dqv).sum(-1, True))
    if head_first:
        dq = q.new_empty(NM, B, H, T, K, dtype=torch.float)
        dk = q.new_empty(NM, B, H, T, K, dtype=torch.float)
        dsk = q.new_empty(NK, B, H, T, M, dtype=torch.float)
    else:
        dq = q.new_empty(NM, B, T, H, K, dtype=torch.float)
        dk = q.new_empty(NM, B, T, H, K, dtype=torch.float)
        dsk = q.new_empty(NK, B, T, H, M, dtype=torch.float)
    gk, gv = None, g
    grid = (NM, NK, N * H)
    fused_recurrent_bwd_kernel[grid](
        q=q,
        k=k,
        v=s,
        g=None,
        gk=gk,
        gv=gv,
        h0=hk0,
        do=dok,
        dq=dq,
        dk=dk,
        dv=dsk,
        dht=dhkt,
        dh0=dhk0,
        offsets=offsets,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=M,
        BK=BK,
        BV=BM,
        USE_G=False,
        USE_GK=False,
        USE_GV=True,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dsk = dsk.sum(0)

    dgv = chunk_global_cumsum(dok.float() * ok.float() - dsk * s.float(),
                              reverse=not reverse,
                              offsets=offsets,
                              head_first=head_first)

    ds = dsk.add_(dsv)
    dg = dgk.add_(dgv)

    return dq, dk, dv, ds, dg, dhk0, dhv0


class FusedRecurrentGSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        scale: Optional[float] = None,
        hk0: Optional[torch.Tensor] = None,
        hv0: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        T = q.shape[2] if head_first else q.shape[1]
        if T == 1 and not q.requires_grad:
            o, (hkt, hvt) = fused_recurrent_gsa_inference(
                q=q,
                k=k,
                v=v,
                s=s,
                g=g,
                initial_state=(hk0, hv0),
                output_final_state=output_final_state,
                scale=scale,
                head_first=head_first
            )
            return o, hkt, hvt
        ok, hkt, qv, ov, hvt = fused_recurrent_gsa_fwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            initial_state=(hk0, hv0),
            output_final_state=output_final_state,
            scale=scale,
            reverse=reverse,
            offsets=offsets,
            head_first=head_first
        )
        ctx.save_for_backward(q, k, v, s, g, qv, hk0, hv0, ok)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.offsets = offsets
        ctx.head_first = head_first
        return ov.to(q.dtype), hkt, hvt

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dhkt=None, dhvt=None):
        q, k, v, s, g, qv, hk0, hv0, ok = ctx.saved_tensors
        scale = ctx.scale
        reverse = ctx.reverse
        offsets = ctx.offsets
        head_first = ctx.head_first

        # not supported yet.
        if dhkt is not None or dhvt is not None:
            if g is not None:
                assert g.requires_grad is False, "Cannot load final state gradient and use gates at the same time"
        dq, dk, dv, ds, dg, dhk0, dhv0 = fused_recurrent_gsa_bwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            qv=qv,
            hk0=hk0,
            hv0=hv0,
            ok=ok,
            do=do,
            dhkt=dhkt,
            dhvt=dhvt,
            scale=scale,
            reverse=reverse,
            offsets=offsets,
            head_first=head_first
        )
        return dq.to(q), dk.to(k), dv.to(v), ds.to(s), dg.to(g), None, dhk0, dhv0, None, None, None, None


def fused_recurrent_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[Tuple[torch.Tensor]] = None,
    output_final_state: Optional[bool] = False,
    reverse: Optional[bool] = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        s (torch.Tensor):
            slot representations of shape `[B, H, T, M]` if `head_first=True` else `[B, T, H, M]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T, M]` applied to keys.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]` and `[N, H, M, V]`.
            Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (Tuple[torch.Tensor]):
            Final state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gsa import fused_recurrent_gsa
        # inputs with equal lengths
        >>> B, T, H, K, V, M = 4, 2048, 4, 512, 512, 64
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> s = torch.randn(B, T, H, M, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, M, device='cuda'))
        >>> h0 = (torch.randn(B, H, K, M, device='cuda'), torch.randn(B, H, M, V, device='cuda'))
        >>> o, (hk, hv) = fused_recurrent_gsa(q, k, v, s, g,
                                              initial_state=h0,
                                              output_final_state=True,
                                              head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, s, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, s, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, (hk_var, hv_var) = fused_recurrent_gsa(q, k, v, s, g,
                                                          initial_state=h0,
                                                          output_final_state=True,
                                                          cu_seqlens=cu_seqlens,
                                                          head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert hk.allclose(hk_var)
        >>> assert hv.allclose(hv_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state[0].shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(cu_seqlens) - 1} rather than {initial_state[0].shape[0]}.")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if initial_state is None:
        initial_state = (None, None)
    o, *final_state = FusedRecurrentGSAFunction.apply(
        q,
        k,
        v,
        s,
        g,
        scale,
        *initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
        head_first
    )
    return o, final_state
