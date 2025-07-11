# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
from einops import rearrange

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.delta_rule.chunk import chunk_delta_rule_bwd
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd as dn_recompute_w_u_fwd
from fla.ops.gated_delta_product.chunk_deltaproduct_h import chunk_gated_delta_product_fwd_h
from fla.ops.gated_delta_product.chunk_deltaproduct_o import chunk_gated_delta_product_fwd_o
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd as gdn_recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_gated_delta_product_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    num_householder: int = 1,
):
    cu_seqlens_dp = cu_seqlens * num_householder if cu_seqlens is not None else None
    if g is not None:
        g_interleaved = g.new_zeros(g.shape[0], g.shape[1], num_householder, g.shape[2], dtype=torch.float32)
        g_interleaved[:, :, 0] = g
        g_interleaved = rearrange(g_interleaved, 'b l n h -> b (l n) h').contiguous()
        g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
        g_interleaved = chunk_local_cumsum(g_interleaved, chunk_size=64, cu_seqlens=cu_seqlens_dp, output_dtype=torch.float32)
    else:
        g_interleaved = None
        g = None
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_interleaved,
        beta=beta,
        cu_seqlens=cu_seqlens_dp,
        output_dtype=torch.float32
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens_dp,
        output_dtype=k.dtype
    )
    if g is not None:
        w, u = gdn_recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g_interleaved,
            cu_seqlens=cu_seqlens_dp,
        )
    else:
        w, u = dn_recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            cu_seqlens=cu_seqlens_dp,
        )
    h, v_new, final_state = chunk_gated_delta_product_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_interleaved,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens_dp,
        num_householder=num_householder,
    )
    o = chunk_gated_delta_product_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        num_householder=num_householder,
    )
    return g, g_interleaved, o, A, final_state


class ChunkGatedDeltaProductFunction(torch.autograd.Function):

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
        scale: float,
        num_householder: int,
        initial_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, g_interleaved, o, A, final_state = chunk_gated_delta_product_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            num_householder=num_householder,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_interleaved, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.num_householder = num_householder
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        q_new = q.new_zeros(q.shape[0], q.shape[1], ctx.num_householder, q.shape[2], q.shape[3])
        q_new[:, :, -1] = q
        do_new = do.new_zeros(do.shape[0], do.shape[1], ctx.num_householder, do.shape[2], do.shape[3])
        do_new[:, :, -1] = do
        q_org, q = q, rearrange(q_new, 'b t n h d -> b (t n) h d')
        do = rearrange(do_new, 'b t n h d -> b (t n) h d')
        # call the gated deltanet kernel for now.
        # TODO: optimize the backward pass like the forward pass.
        if g is not None:
            dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A=A,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=dht,
                cu_seqlens=cu_seqlens * ctx.num_householder if cu_seqlens is not None else None,
            )
            dg = rearrange(dg, 'b (l n) h  -> b l n h ', n=ctx.num_householder)[:, :, 0].contiguous().to(g)
        else:
            dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(
                q=q,
                k=k,
                v=v,
                beta=beta,
                A=A,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=dht,
                cu_seqlens=cu_seqlens * ctx.num_householder if cu_seqlens is not None else None,
            )
            dg = None
        dq = rearrange(dq, 'b (l n) h d -> b l n h d', n=ctx.num_householder)[:, :, -1].contiguous()
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q_org, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg, db.to(beta), None, None, dh0, None, None, None


@torch.compiler.disable
def chunk_gated_delta_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        num_householder (int):
            Number of householder transformations to apply. Default: `1`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use qk l2norm within the kernel for saving GPU memory.
            Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_product
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_product(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_product(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype != torch.float32, "ChunkGatedDeltaProductFunction does not support float32. Please use bfloat16."
    B, T, H, K, V = *q.shape, v.shape[-1]
    assert k.shape == (B, T*num_householder, H, K)
    assert v.shape == (B, T*num_householder, H, V)
    assert beta.shape == (B, T*num_householder, H)
    if g is not None:
        assert g.shape == (B, T, H)

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
    o, final_state = ChunkGatedDeltaProductFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        num_householder,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
    return o, final_state
