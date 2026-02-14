# # Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.gated_oja_rule.chunk_h import chunk_oja_bwd_dhu, chunk_oja_bwd_dvwg_h, chunk_oja_fwd_h
from fla.ops.gated_oja_rule.chunk_kkt import chunk_scaled_dot_kkt_bwd_gk, chunk_scaled_dot_kkt_fwd
from fla.ops.gated_oja_rule.chunk_o import (
    chunk_oja_bwd_dA,
    chunk_oja_bwd_dqk,
    chunk_oja_bwd_dv_o,
    chunk_oja_fwd_o,
)
from fla.ops.gated_oja_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_oja_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    g_cumsum: bool = True,
    cu_seqlens: torch.LongTensor | None = None
):
    if g_cumsum:
        gv = chunk_local_cumsum(gv, chunk_size=64, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k=v,
        gk=gv,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
    w, u, vg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gv=gv,
        cu_seqlens=cu_seqlens,
    )
    h, k_new, final_state = chunk_oja_fwd_h(
        v=vg,
        w=w,
        u=u,
        gv=gv,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    _, o = chunk_oja_fwd_o(
        q=q,
        k=k_new,
        v=v,
        h=h,
        gv=gv,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return gv, o, A, final_state


def chunk_oja_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dgk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    w, u, vg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gv=gv,
        cu_seqlens=cu_seqlens,
    )

    h, k_new, _ = chunk_oja_fwd_h(
        v=vg,
        w=w,
        u=u,
        gv=gv,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )

    dAqk = chunk_oja_bwd_dA(
        v=v,
        gv=gv,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    Aqk, dq, dk_new = chunk_oja_bwd_dqk(
        q=q,
        k=k_new,
        h=h,
        gv=gv,
        dA=dAqk,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    dh, dh0, dk_new = chunk_oja_bwd_dhu(
        q=q,
        vg=vg,
        w=w,
        gv=gv,
        h0=initial_state,
        dht=dht,
        do=do,
        dk=dk_new,
        scale=scale,
        cu_seqlens=cu_seqlens,
        states_in_fp32=False,
    )

    # grid = (NV, NT, B * H)
    dv, dw, dgv_last = chunk_oja_bwd_dvwg_h(
        k=k_new,
        v=v,
        gv=gv,
        h=h,
        dh=dh,
        dk=dk_new,
        dgk=dgk,
        cu_seqlens=cu_seqlens,
    )

    dv, dgv1 = chunk_oja_bwd_dv_o(
        v=v,
        gv=gv,
        o=o,
        A=Aqk,
        dv=dv,
        do=do,
        cu_seqlens=cu_seqlens,
    )

    dk, dv1, db, dgv2, dAvv = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        gv=gv,
        A=A,
        dw=dw,
        du=dk_new,
        cu_seqlens=cu_seqlens,
    )

    dv2, dgv3, db2 = chunk_scaled_dot_kkt_bwd_gk(
        k=v,
        g=gv,
        beta=beta,
        dA=dAvv,
        cu_seqlens=cu_seqlens,
    )

    dv = dv.add_(dv1).add_(dv2)
    db = db.add_(db2)
    dgv = dgv_last.add_(chunk_local_cumsum(dgv1.add_(dgv2).add_(dgv3), chunk_size=64, reverse=True, cu_seqlens=cu_seqlens))
    return dq, dk, dv, db, dgv, dh0


class ChunkOJAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gv: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_q_l2norm: bool = False,
        use_k_l2norm: bool = False,
    ):
        q_rstd, k_rstd = None, None
        if use_q_l2norm:
            q, q_rstd = l2norm_fwd(q)
        if use_k_l2norm:
            k, k_rstd = l2norm_fwd(k)

        gv, o, A, final_state = chunk_oja_fwd(
            q=q,
            k=k,
            v=v,
            gv=gv,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, gv, beta, A, o, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_q_l2norm = use_q_l2norm
        ctx.use_k_l2norm = use_k_l2norm
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, q_rstd, k, k_rstd, v, gv, beta, A, o, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_oja_bwd(
            q=q,
            k=k,
            v=v,
            gv=gv,
            beta=beta,
            A=A,
            o=o,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
        )

        if ctx.use_q_l2norm:
            dq = l2norm_bwd(q, q_rstd, dq)
        if ctx.use_k_l2norm:
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(gv), db.to(beta), None, dh0, None, None, None, None


@torch.compiler.disable
def chunk_gated_oja_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_q_l2norm: bool = False,
    use_k_l2norm: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
):
    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if 'use_qk_l2norm_in_kernel' in kwargs and (not use_q_l2norm and not use_k_l2norm):
        use_q_l2norm = True
        use_k_l2norm = True

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkOJAFunction.apply(
        q,
        k,
        v,
        gv,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_q_l2norm,
        use_k_l2norm
    )
    return o, final_state
