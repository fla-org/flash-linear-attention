# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.delta_rule.wy_fast import prepare_wy_repr_bwd, prepare_wy_repr_fwd, recompute_w_u_fwd
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_mode_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    """Forward pass for MomentumDeltaNet chunk mode."""
    w, u, A = prepare_wy_repr_fwd(
        k=k,
        v=v,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=None,
        scale=1.0 / math.sqrt(q.shape[-1]),
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    return o, A, final_state


def chunk_mode_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    """Backward pass for MomentumDeltaNet chunk mode."""
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        do=do,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=None,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    dq, dk, dw, _ = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        h=h,
        w=w,
        dv=dv,
        do=do,
        dh=dh,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_size=chunk_size,
    )
    dk2, dv, db = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
    )
    dk.add_(dk2)
    return dq, dk, dv, db, dh0


class ChunkModeRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_size: int = 64,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        chunk_indices = prepare_chunk_indices(
            cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
        o, A, final_state = chunk_mode_rule_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens, chunk_indices)
        ctx.scale = 1.0 / math.sqrt(q.shape[-1])
        ctx.chunk_size = chunk_size
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors

        dq, dk, dv, db, dh0 = chunk_mode_rule_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), None, dh0, None, None, None, None, None


@torch.compiler.disable
def chunk_mode_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
):
    r"""MomentumDeltaNet chunk mode rule.

    Args:
        q (torch.Tensor): queries of shape `[B, T, H, K]`.
        k (torch.Tensor): keys of shape `[B, T, H, K]`.
        v (torch.Tensor): values of shape `[B, T, H, V]`.
        beta (torch.Tensor): betas of shape `[B, T, H]`.
        initial_state (Optional[torch.Tensor]): Initial state of shape `[N, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]): Whether to output the final state. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]): Whether to use qk l2norm in kernel. Default: `False`.
        cu_seqlens (torch.LongTensor): Cumulative sequence lengths for variable-length training.
        cu_seqlens_cpu (torch.LongTensor): CPU copy of cu_seqlens. Default: `None`.
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkModeRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, num of head, seq len)."

    chunk_size = kwargs.pop('chunk_size', 64)
    if chunk_size not in (16, 32, 64):
        raise ValueError(f"`chunk_size` must be 16, 32, or 64, got {chunk_size}.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    o, final_state = ChunkModeRuleFunction.apply(
        q,
        k,
        v,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_size,
    )
    return o, final_state
