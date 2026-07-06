# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.atk.chunk_atk_bwd import chunk_atk_bwd
from fla.ops.atk.chunk_atk_fwd import chunk_atk_fwd, recompute_atk_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.cp import FLACPContext
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.precond_gated_delta_rule.chunk_precond_kkt_fwd import chunk_precond_kkt_fwd
from fla.ops.precond_gated_delta_rule.chunk_precond_wy_bwd import prepare_precond_wy_repr_bwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_precond_gated_delta_rule_fwd(
    q: torch.Tensor,
    k_read: torch.Tensor,
    k_write: torch.Tensor,
    v: torch.Tensor,
    g_atk: torch.Tensor,
    g: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_A_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    transpose_state_layout: bool = False,
):
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    k_precond, at = chunk_atk_fwd(
        k=k_write,
        beta=beta_atk,
        log_g=g_atk,
        chunk_size=64,
        initial_A_state=initial_A_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        x=x,
        eps=eps,
        log_atk_scale=log_atk_scale,
    )

    g = chunk_local_cumsum(
        g,
        chunk_size=64,
        scale=RCP_LN2 if use_exp2 else None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    # Asymmetric KKT: uses k_read for correction, k_precond for write
    A = chunk_precond_kkt_fwd(
        k=k_read,
        k_precond=k_precond,
        g=g,
        beta=beta,
        chunk_size=64,
        output_dtype=torch.float32,
        cu_seqlens=cu_seqlens,
        use_exp2=use_exp2,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=k_read.dtype,
    )
    # obtain WY representation. u is actually the new v.
    w, u = recompute_w_u_fwd(
        k=k_read,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k_precond,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=transpose_state_layout,
    )

    o = chunk_fwd_o(
        q=q,
        k=k_precond,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=transpose_state_layout,
    )

    return g, o, A, final_state, at


def chunk_precond_gated_delta_rule_bwd(
    q: torch.Tensor,
    k_read: torch.Tensor,
    k_write: torch.Tensor,
    v: torch.Tensor,
    g_atk: torch.Tensor,
    g: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_A_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    transpose_state_layout: bool = False,
):
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    # Recompute ATK forward intermediates
    k_precond, ac_atk, a_atk, sa_atk = recompute_atk_fwd(
        k=k_write,
        beta=beta_atk,
        log_g=g_atk,
        chunk_size=64,
        initial_A_state=initial_A_state,
        cu_seqlens=cu_seqlens,
        x=x,
        eps=eps,
        log_atk_scale=log_atk_scale,
    )

    w, u = recompute_w_u_fwd(
        k=k_read,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k_precond,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=transpose_state_layout,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k_precond,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k_precond,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=transpose_state_layout,
    )
    dq, dk_precond, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k_precond,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=transpose_state_layout,
    )
    # Asymmetric WY backward: produces separate grads for k_read and k_precond
    dk_read_wy, dk_precond_wy, dv, dbeta_wy, dg2 = prepare_precond_wy_repr_bwd(
        k=k_read,
        k_precond=k_precond,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )

    dbeta = dbeta_wy
    dk_precond.add_(dk_precond_wy)

    # ATK backward (uses precomputed forward intermediates)
    dk_write_atk, dbeta_atk, dg_atk, d_log_atk_scale, dh0_atk = chunk_atk_bwd(
        k=k_write,
        g_raw=g_atk,
        beta=beta_atk,
        dk_precond=dk_precond,
        ac=ac_atk,
        a=a_atk,
        sa=sa_atk,
        initial_A_state=initial_A_state,
        cu_seqlens=cu_seqlens,
        x=x,
        eps=eps,
        log_atk_scale=log_atk_scale,
    )

    dk_read = dk_read_wy
    dk_write = dk_write_atk

    dg.add_(dg2)
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)

    return dq, dk_read, dk_write, dv, dbeta_atk, dbeta, dg_atk, dg, dh0, dh0_atk, d_log_atk_scale


class ChunkPrecondGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_atk: torch.Tensor,
        g: torch.Tensor,
        beta_atk: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        initial_A_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        transpose_state_layout: bool = False,
        x: float = 1.5,
        eps: float = 1e-6,
        log_atk_scale: torch.Tensor = None,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        k_read = k
        k_write = k

        H = k.shape[2]
        log_atk_scale_was_tensor = isinstance(log_atk_scale, torch.Tensor)

        if log_atk_scale is None:
            log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32, device=k.device)

        chunk_indices = prepare_chunk_indices(
            cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None

        g, o, A, h_final, at = chunk_precond_gated_delta_rule_fwd(
            q=q,
            k_read=k_read,
            k_write=k_write,
            v=v,
            g_atk=g_atk,
            g=g,
            beta_atk=beta_atk,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_A_state=initial_A_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
            chunk_indices=chunk_indices,
            use_exp2=True,
            x=x,
            eps=eps,
            log_atk_scale=log_atk_scale,
            transpose_state_layout=transpose_state_layout,
        )

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v,
            g_atk, g,
            beta_atk, beta,
            A, initial_state, cu_seqlens,
            log_atk_scale, chunk_indices
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.initial_A_state = initial_A_state
        ctx.cp_context = cp_context
        ctx.transpose_state_layout = transpose_state_layout
        ctx.x = x
        ctx.eps = eps
        ctx.log_atk_scale_was_tensor = log_atk_scale_was_tensor

        return o.to(q.dtype), h_final, at

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, dat):
        (q, q_rstd, k, k_rstd, v,
         g_atk, g,
         beta_atk, beta,
         A, initial_state, cu_seqlens,
         log_atk_scale, chunk_indices) = ctx.saved_tensors

        k_read = k
        k_write = k

        dq, dk_read, dk_write, dv, dbeta_atk, dbeta, dg_atk, dg, dh0, dh0_atk, d_log_atk_scale = chunk_precond_gated_delta_rule_bwd(
            q=q,
            k_read=k_read,
            k_write=k_write,
            v=v,
            g_atk=g_atk,
            g=g,
            beta_atk=beta_atk,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            initial_A_state=ctx.initial_A_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            cp_context=ctx.cp_context,
            chunk_indices=chunk_indices,
            use_exp2=True,
            x=ctx.x,
            eps=ctx.eps,
            log_atk_scale=log_atk_scale,
            transpose_state_layout=ctx.transpose_state_layout,
        )

        dk_read.add_(dk_write)
        dk = dk_read

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        d_log_atk_scale_out = d_log_atk_scale.to(log_atk_scale) if ctx.log_atk_scale_was_tensor else None
        return (
            dq.to(q), dk.to(k), dv.to(v),
            dg_atk.to(g_atk), dg.to(g),
            dbeta_atk.to(beta_atk), dbeta.to(beta),
            None, dh0, dh0_atk, None,  # scale, initial_state, initial_A_state, output_final_state
            None, None, None, None, None,  # use_qk_l2norm_in_kernel, cu_seqlens, cu_seqlens_cpu, cp_context, transpose_state_layout
            None, None,  # x, eps
            d_log_atk_scale_out,  # log_atk_scale
        )


@torch.compiler.disable
def chunk_precond_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_atk: torch.Tensor,
    g: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_A_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    **kwargs,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA (Grouped Value Attention) is applied if `HV > H`, where `HV` must be divisible by `H`.
        g_atk (torch.Tensor):
            ATK gates (decays) in log space of shape `[B, T, H]`.
        g (torch.Tensor):
            gates (decays) in log space of shape `[B, T, HV]`.
        beta_atk (torch.Tensor):
            ATK betas of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial recurrent state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        initial_A_state (Optional[torch.Tensor]):
            Initial ATK diagonal state of shape `[N, H, K]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cp_context (Optional[FLACPContext]):
            Context parallel context for distributed training across multiple devices.
            When provided, `initial_state` and `output_final_state` are not supported,
            and `cu_seqlens` will be overridden by the context. Default: `None`.
        transpose_state_layout (Optional[bool]):
            Whether to use the transposed state layout for the hidden state.
            Default: `False`.
        x (float):
            Squash range parameter. Default: `1.5`.
        eps (float):
            Epsilon for numerical stability in the squash function. Default: `1e-6`.
        log_atk_scale (Optional[torch.Tensor]):
            Per-head log-space center of shape `[H]`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        recurrent_state (torch.Tensor):
            Final recurrent state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
        A_state (torch.Tensor):
            Final ATK diagonal state of shape `[N, H, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.precond_gated_delta_rule import chunk_precond_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta_atk = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g_atk = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht, at = chunk_precond_gated_delta_rule(
            q, k, v, g_atk, g, beta_atk, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta_atk, beta, g_atk, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta_atk, beta, g_atk, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht, at = chunk_precond_gated_delta_rule(
            q, k, v, g_atk, g, beta_atk, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    # Validate head dimensions
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            f"q and k must have the same number of heads, "
            f"but got q.shape[2]={q.shape[2]} and k.shape[2]={k.shape[2]}"
        )
    H, HV = q.shape[2], v.shape[2]
    if HV % H != 0:
        raise ValueError(
            f"For GVA, num_v_heads (HV={HV}) must be evenly divisible by "
            f"num_heads (H={H}), but got HV % H = {HV % H}"
        )

    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )

    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        cu_seqlens = cp_context.cu_seqlens
        if cp_context.cu_seqlens_cpu is not None:
            cu_seqlens_cpu = cp_context.cu_seqlens_cpu

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
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, h_final, a_final = ChunkPrecondGatedDeltaRuleFunction.apply(
        q, k, v,
        g_atk, g,
        beta_atk, beta,
        scale,
        initial_state,
        initial_A_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        cp_context,
        transpose_state_layout,
        x,
        eps,
        log_atk_scale,
    )

    return o, h_final, a_final
