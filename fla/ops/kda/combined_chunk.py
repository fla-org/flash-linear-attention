# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch
from einops import rearrange

from fla.modules.convolution import causal_conv1d_bwd, causal_conv1d_fwd
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.kda.chunk import chunk_kda_bwd, chunk_kda_fwd
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class CombinedChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q_conv1d_weight: torch.Tensor,
        k_conv1d_weight: torch.Tensor,
        v_conv1d_weight: torch.Tensor,
        head_dim: int,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        conv1d_activation: str = 'silu',
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ):
        # conv1d forward (saves q, k, v pre-conv instead of post-conv to save 3 activations)
        q_conv, _ = causal_conv1d_fwd(
            x=q, weight=q_conv1d_weight, bias=None, residual=None,
            activation=conv1d_activation, cu_seqlens=cu_seqlens,
        )
        k_conv, _ = causal_conv1d_fwd(
            x=k, weight=k_conv1d_weight, bias=None, residual=None,
            activation=conv1d_activation, cu_seqlens=cu_seqlens,
        )
        v_conv, _ = causal_conv1d_fwd(
            x=v, weight=v_conv1d_weight, bias=None, residual=None,
            activation=conv1d_activation, cu_seqlens=cu_seqlens,
        )

        q_conv = rearrange(q_conv, '... (h d) -> ... h d', d=head_dim)
        k_conv = rearrange(k_conv, '... (h d) -> ... h d', d=head_dim)
        v_conv = rearrange(v_conv, '... (h d) -> ... h d', d=head_dim)
        g = rearrange(g, '... (h d) -> ... h d', d=head_dim)

        g_org = None
        if use_gate_in_kernel:
            g_org = g
            g = kda_gate_fwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
            )
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q_conv, q_rstd = l2norm_fwd(q_conv)
            k_conv, k_rstd = l2norm_fwd(k_conv)

        chunk_size = 64
        g = chunk_local_cumsum(
            g=g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices
        )
        o, Aqk, Akk, final_state = chunk_kda_fwd(
            q=q_conv,
            k=k_conv,
            v=v_conv,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        if use_gate_in_kernel:
            g = None

        ctx.save_for_backward(
            q, k, v, q_conv1d_weight, k_conv1d_weight, v_conv1d_weight,
            q_rstd, k_rstd, g, g_org, beta, A_log, dt_bias, Aqk, Akk,
            initial_state, cu_seqlens, chunk_indices
        )
        ctx.head_dim = head_dim
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.conv1d_activation = conv1d_activation
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        (
            q, k, v, q_conv1d_weight, k_conv1d_weight, v_conv1d_weight,
            q_rstd, k_rstd, g, g_org, beta, A_log, dt_bias, Aqk, Akk,
            initial_state, cu_seqlens, chunk_indices
        ) = ctx.saved_tensors

        q_conv, _ = causal_conv1d_fwd(
            x=q, weight=q_conv1d_weight, bias=None, residual=None,
            activation=ctx.conv1d_activation, cu_seqlens=cu_seqlens,
        )
        k_conv, _ = causal_conv1d_fwd(
            x=k, weight=k_conv1d_weight, bias=None, residual=None,
            activation=ctx.conv1d_activation, cu_seqlens=cu_seqlens,
        )
        v_conv, _ = causal_conv1d_fwd(
            x=v, weight=v_conv1d_weight, bias=None, residual=None,
            activation=ctx.conv1d_activation, cu_seqlens=cu_seqlens,
        )

        q_conv = rearrange(q_conv, '... (h d) -> ... h d', d=ctx.head_dim)
        k_conv = rearrange(k_conv, '... (h d) -> ... h d', d=ctx.head_dim)
        v_conv = rearrange(v_conv, '... (h d) -> ... h d', d=ctx.head_dim)

        if ctx.use_gate_in_kernel:
            g = kda_gate_fwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
            )
            g = chunk_local_cumsum(
                g=g,
                chunk_size=ctx.chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices
            )
        if ctx.use_qk_l2norm_in_kernel:
            q_conv, _ = l2norm_fwd(q_conv)
            k_conv, _ = l2norm_fwd(k_conv)

        dq, dk, dv, db, dg, dh0 = chunk_kda_bwd(
            q=q_conv,
            k=k_conv,
            v=v_conv,
            g=g,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q_conv, q_rstd, dq)
            dk = l2norm_bwd(k_conv, k_rstd, dk)
        dA, dbias = None, None
        if ctx.use_gate_in_kernel:
            dg, dA, dbias = kda_gate_bwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                dyg=dg,
                dyb=db,
            )
            dA = dA.to(A_log)
            if dt_bias is not None:
                dbias = dbias.to(dt_bias)

        dq = rearrange(dq, '... h d -> ... (h d)')
        dk = rearrange(dk, '... h d -> ... (h d)')
        dv = rearrange(dv, '... h d -> ... (h d)')
        dg = rearrange(dg, '... h d -> ... (h d)')

        dq, dq_w, _, _, _ = causal_conv1d_bwd(
            x=q, dy=dq, dht=None, weight=q_conv1d_weight,
            activation=ctx.conv1d_activation, cu_seqlens=cu_seqlens,
        )
        dk, dk_w, _, _, _ = causal_conv1d_bwd(
            x=k, dy=dk, dht=None, weight=k_conv1d_weight,
            activation=ctx.conv1d_activation, cu_seqlens=cu_seqlens,
        )
        dv, dv_w, _, _, _ = causal_conv1d_bwd(
            x=v, dy=dv, dht=None, weight=v_conv1d_weight,
            activation=ctx.conv1d_activation, cu_seqlens=cu_seqlens,
        )

        return (
            dq.to(q), dk.to(k), dv.to(v), dg.to(g_org if g_org is not None else g), db.to(beta),
            dA, dbias, dq_w, dk_w, dv_w,
            None, None, dh0, None, None, None, None, None, None
        )


@torch.compiler.disable
def combined_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    q_conv1d_weight: torch.Tensor,
    k_conv1d_weight: torch.Tensor,
    v_conv1d_weight: torch.Tensor,
    head_dim: int,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    conv1d_activation: str = 'silu',
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    **kwargs,
):
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    if scale is None:
        scale = head_dim ** -0.5
    o, final_state = CombinedChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        q_conv1d_weight,
        k_conv1d_weight,
        v_conv1d_weight,
        head_dim,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        conv1d_activation,
        cu_seqlens,
        chunk_indices,
    )
    return o, final_state
