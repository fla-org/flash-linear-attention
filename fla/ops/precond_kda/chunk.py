# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Related files are modified and supported by the Moonshot AI Team

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.atk.chunk_atk_bwd import chunk_atk_bwd
from fla.ops.atk.chunk_atk_fwd import chunk_atk_fwd, recompute_atk_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.cp import FLACPContext
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from fla.ops.precond_kda.chunk_bwd import chunk_precond_kda_bwd_dAv, chunk_precond_kda_bwd_wy_dqkg
from fla.ops.precond_kda.chunk_intra import chunk_precond_kda_bwd_intra, chunk_precond_kda_fwd_intra
from fla.ops.precond_kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_precond_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,           # Diagonal gate for KDA [B, T, H, K] - ALREADY cumsummed
    g_atk: torch.Tensor,       # Scalar gate for ATK [B, T, H]
    beta_atk: torch.Tensor,    # Beta for ATK [B, T, H]
    beta: torch.Tensor,    # Beta for KDA [B, T, H]
    scale: float,
    initial_state: torch.Tensor,
    initial_A_state: torch.Tensor,
    output_final_state: bool,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    solve_tril_precision: str | None = None,
    safe_gate: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    transpose_state_layout: bool = False,
):
    """
    Forward pass for preconditioned KDA.

    Args:
        q: [B, T, H, K] - queries
        k: [B, T, H, K] - keys
        v: [B, T, H, V] - values
        g: [B, T, H, K] - diagonal gate for KDA (chunk-local cumsum, log2 space)
        g_atk: [B, T, H] - scalar gate for ATK (log space)
        beta_atk: [B, T, H] - beta for ATK
        beta: [B, T, H] - beta for KDA
        scale: attention scale
        initial_state: [B, H, K, V] or None
        output_final_state: bool
        x: range parameter for symmetric squash. M in [1/x, x]
        eps: epsilon for log safety
        log_atk_scale: per-head log-space center [H]

    Returns:
        o, Aqk, Akk, final_state, at
    """
    B, T, H, K = k.shape

    # Step 1: ATK preconditioning with symmetric fast squash
    # Only need k_precond and at (final state); ac/a_atk/sa_atk recomputed in backward
    k_precond, at = chunk_atk_fwd(
        k=k,
        beta=beta_atk,
        log_g=g_atk,
        chunk_size=chunk_size,
        initial_A_state=initial_A_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        x=x,
        eps=eps,
        log_atk_scale=log_atk_scale,
    )

    # Step 2: KDA intra-chunk computation (asymmetric)
    # Aqk = q @ k_precond^T, Akk = k @ k_precond^T
    w, u, kg, Aqk, Akk = chunk_precond_kda_fwd_intra(
        q=q,
        k=k,           # Original k for Akk row side, w computation
        k_precond=k_precond,  # k_precond for column side, kg
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        solve_tril_precision=solve_tril_precision,
        safe_gate=safe_gate,
    )

    # Step 3: Hidden state update (uses kg which is k_precond gated)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,  # k_precond gated for write
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=transpose_state_layout,
    )

    # Step 4: Output computation
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=transpose_state_layout,
    )

    return o, Aqk, Akk, final_state, at, w, u, kg, v_new, h


def chunk_precond_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,            # Gate cumsum (log2 space)
    g_atk: torch.Tensor,        # Raw ATK gate
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_A_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    transpose_state_layout: bool = False,
    safe_gate: bool = False,
    disable_recompute: bool = False,
    w: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
    kg: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
    h: torch.Tensor | None = None,
):
    """
    Backward pass for preconditioned KDA with symmetric fast squash preconditioning.

    Returns:
        dq, dk, dv, dg, dg_atk, dbeta_atk, dbeta, d_log_atk_scale, dh0
    """
    # Recompute ATK forward intermediates (k_precond, ac, a_atk, sa_atk)
    k_precond, ac, a_atk, sa_atk = recompute_atk_fwd(
        k=k,
        beta=beta_atk,
        log_g=g_atk,
        chunk_size=chunk_size,
        initial_A_state=initial_A_state,
        cu_seqlens=cu_seqlens,
        x=x,
        eps=eps,
        log_atk_scale=log_atk_scale,
    )

    if not disable_recompute:
        # Step 1: Recompute WY representation (asymmetric)
        w, u, qg, kg = recompute_w_u_fwd(
            k=k,
            k_precond=k_precond,
            v=v,
            beta=beta,
            A=Akk,
            gk=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            q=q,
            output_qg=True,
            output_kg=True,
        )

        # Step 2: Recompute hidden state (uses kg = gated k_precond)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            state_v_first=transpose_state_layout,
        )
    else:
        # Intermediates (w, u, kg, v_new, h) saved from forward; only need qg
        _, _, qg, _ = recompute_w_u_fwd(
            k=k,
            k_precond=k_precond,
            v=v,
            beta=beta,
            A=Akk,
            gk=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            q=q,
            output_qg=True,
            output_kg=False,
        )

    BT = chunk_size

    # Step 3: dAqk and local dv (matching KDA's chunk_kda_bwd_dAv)
    dAqk, dv = chunk_precond_kda_bwd_dAv(
        q=q,
        k=k,
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        chunk_indices=chunk_indices,
    )

    # Step 4: Backward through hidden state
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=transpose_state_layout,
    )

    # Step 5+6: Inter-chunk + WY backward (matching KDA's chunk_kda_bwd_wy_dqkg_fused)
    dq, dk, dkg, dv, dbeta, dg, dAkk = chunk_precond_kda_bwd_wy_dqkg(
        q=q,
        k=k,
        k_precond=k_precond,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )

    # Step 7: Asymmetric intra backward
    dq2, dk_intra, dk_precond2, dbeta2, dg3 = chunk_precond_kda_bwd_intra(
        q=q,
        k=k,
        k_precond=k_precond,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        dk_precond=dkg,
        db=dbeta,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
    )

    dk_precond_total = dk_precond2
    dk = dk_intra

    dk_atk, dbeta_atk_grad, dg_atk, d_log_atk_scale, dh0_atk = chunk_atk_bwd(
        k=k,
        g_raw=g_atk,
        beta=beta_atk,
        ac=ac,
        a=a_atk,
        sa=sa_atk,
        dk_precond=dk_precond_total,
        initial_A_state=initial_A_state,
        cu_seqlens=cu_seqlens,
        x=x,
        eps=eps,
        log_atk_scale=log_atk_scale,
    )

    # Combine dk gradients
    dk_total = dk + dk_atk

    # dg3 from asymmetric intra backward already has reverse cumsum applied
    dg_total = dg3

    return dq2, dk_total, dv, dg_total, dg_atk, dbeta_atk_grad, dbeta2, d_log_atk_scale, dh0, dh0_atk


class ChunkPrecondKDAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        g_atk: torch.Tensor,
        beta_atk: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        initial_A_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        safe_gate: bool = False,
        cp_context: FLACPContext | None = None,
        transpose_state_layout: bool = False,
        x: float = 1.5,
        eps: float = 1e-6,
        log_atk_scale: torch.Tensor = None,
        lower_bound: float | None = None,
        solve_tril_precision: str | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
    ):
        chunk_size = 64

        # Apply L2 normalization
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        # Compute gate + cumsum
        g_org = None
        if use_gate_in_kernel:
            g_org = g
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                dt_bias=dt_bias,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
            )
        else:
            g = chunk_local_cumsum(
                g=g,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

        o, Aqk, Akk, final_state, at, w, u, kg, v_new, h = chunk_precond_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_atk=g_atk,
            beta_atk=beta_atk,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_A_state=initial_A_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
            chunk_indices=chunk_indices,
            solve_tril_precision=solve_tril_precision,
            safe_gate=safe_gate,
            x=x,
            eps=eps,
            log_atk_scale=log_atk_scale,
            transpose_state_layout=transpose_state_layout,
        )

        if return_intermediate_states:
            assert torch.is_inference_mode_enabled(), "return_intermediate_states is only allowed in inference mode"
            assert disable_recompute is False, "return_intermediate_states must be used with disable_recompute=False"
            return o.to(q.dtype), final_state, at, h

        # Don't save computed g when use_gate_in_kernel (will recompute in backward)
        if use_gate_in_kernel:
            g = None

        # When disable_recompute=False (default), delete intermediates to save memory
        if not disable_recompute:
            w, u, kg, v_new, h = None, None, None, None, None

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, g_org, g_atk, beta_atk, beta,
            Aqk, Akk, initial_state,
            A_log, dt_bias, log_atk_scale, cu_seqlens, chunk_indices,
            w, u, kg, v_new, h
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.safe_gate = safe_gate
        ctx.chunk_size = chunk_size
        ctx.initial_A_state = initial_A_state
        ctx.cp_context = cp_context
        ctx.transpose_state_layout = transpose_state_layout
        ctx.x = x
        ctx.eps = eps
        ctx.lower_bound = lower_bound
        ctx.disable_recompute = disable_recompute

        return o.to(q.dtype), final_state, at

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, dat):
        (q, q_rstd, k, k_rstd, v, g, g_org, g_atk, beta_atk, beta,
         Aqk, Akk, initial_state,
         A_log, dt_bias, log_atk_scale, cu_seqlens, chunk_indices,
         w, u, kg, v_new, h) = ctx.saved_tensors

        # Recompute g (cumsummed) if use_gate_in_kernel was used
        if ctx.use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                chunk_size=ctx.chunk_size,
                scale=RCP_LN2,
                dt_bias=dt_bias,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=ctx.lower_bound,
            )

        dq, dk, dv, dg, dg_atk, dbeta_atk, dbeta, d_log_atk_scale, dh0, dh0_atk = chunk_precond_kda_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_atk=g_atk,
            beta_atk=beta_atk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=initial_state,
            initial_A_state=ctx.initial_A_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            cp_context=ctx.cp_context,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
            x=ctx.x,
            eps=ctx.eps,
            log_atk_scale=log_atk_scale,
            transpose_state_layout=ctx.transpose_state_layout,
            safe_gate=ctx.safe_gate,
            disable_recompute=ctx.disable_recompute,
            w=w,
            u=u,
            kg=kg,
            v_new=v_new,
            h=h,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # Compute gradients for A_log and dt_bias if use_gate_in_kernel
        dA_log, ddt_bias = None, None
        if ctx.use_gate_in_kernel:
            dg, dA_log, ddt_bias = kda_gate_bwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                dyg=dg,
                lower_bound=ctx.lower_bound,
            )
            dA_log = dA_log.to(A_log)
            if dt_bias is not None:
                ddt_bias = ddt_bias.to(dt_bias)

        # For return, use g_org if use_gate_in_kernel, else g
        g_for_dtype = g_org if ctx.use_gate_in_kernel else g

        # Format d_log_atk_scale for return
        if log_atk_scale is not None and d_log_atk_scale is not None:
            d_log_atk_scale = d_log_atk_scale.to(log_atk_scale)
        else:
            d_log_atk_scale = None

        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g_for_dtype),
            dg_atk.to(g_atk),
            dbeta_atk.to(beta_atk),
            dbeta.to(beta),
            None,  # scale
            dh0,   # initial_state
            dh0_atk,  # initial_A_state
            None,  # output_final_state
            None,  # use_qk_l2norm_in_kernel
            None,  # use_gate_in_kernel
            dA_log,  # A_log
            ddt_bias,  # dt_bias
            None,  # cu_seqlens
            None,  # cu_seqlens_cpu
            None,  # chunk_indices
            None,  # safe_gate
            None,  # cp_context
            None,  # transpose_state_layout
            None,  # x
            None,  # eps
            d_log_atk_scale,  # log_atk_scale
            None,  # lower_bound
            None,  # solve_tril_precision
            None,  # disable_recompute
            None,  # return_intermediate_states
        )


@torch.compiler.disable
def chunk_precond_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_A_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    solve_tril_precision: str | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    **kwargs,
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
            (forget) gating tensor (in log space!) of shape `[B, T, H, K]`.
        g_atk (torch.Tensor):
            ATK gates (decays) in log space of shape `[B, T, H]`.
        beta_atk (torch.Tensor):
            ATK betas of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        initial_A_state (Optional[torch.Tensor]):
            Initial ATK diagonal state of shape `[N, H, K]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space KDA decay internally.
            - If `True`:
              The passed `g` acts as the raw input for `-exp(A_log).view(H, -1) * softplus(g + dt_bias.view(H, K))`.
              Note that as part of the input arguments,
              `A_log` (shape `[H]`) and the optional `dt_bias` (shape `[H * K]`) should be provided.
            - If `False`, `g` is expected to be the pre-computed decay value.
            Default: `False`.
        safe_gate (bool):
            Whether the kernel can assume the gate values (in log space) are in a safe range
            and use M=16 TensorCore acceleration for higher throughput.
            The safe range is ``[lower_bound, 0)``. With the default ``lower_bound=-5``,
            the per-step decay factor ``exp(g)`` is bounded in ``[exp(-5), 1) ≈ [0.0067, 1)``,
            meaning each step retains at least ~0.67% of the state -- a negligible loss that
            has minimal impact on model quality while enabling significant speedup.
            Requires ``lower_bound`` to be set. Default: ``False``.
        lower_bound (Optional[float]):
            Lower bound for the forget gate (in log space) when ``use_gate_in_kernel=True``.
            Changes the gate activation from ``-exp(A_log) * softplus(g + dt_bias)``
            to ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``,
            which naturally clamps the output to ``[lower_bound, 0)``.
            Recommended value: ``-5`` (i.e., ``exp(-5) ≈ 0.0067``). Default: ``None``.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cu_seqlens_cpu (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        disable_recompute (bool):
            Whether to disable gradient recomputation in the kernel. When `True`, the kernel
            will save all intermediate activations for backward pass, which is beneficial
            for training small models at the cost of increased memory usage. Default: `False`.
        return_intermediate_states (bool):
            If True, returns intermediate state `h` for inference scenarios (e.g., vLLM).
            Must be used within `torch.inference_mode()` and will return a 4-tuple instead of 3-tuple.
            This is not intended for training as it bypasses autograd. Default: `False`.
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
        - Normal mode (return_intermediate_states=False): A tuple (o, final_state, A_state)
            o (torch.Tensor):
                Outputs of shape `[B, T, H, V]`.
            final_state (torch.Tensor):
                Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
            A_state (torch.Tensor):
                Final ATK diagonal state of shape `[N, H, K]` if `output_final_state=True` else `None`.
        - Inference mode (return_intermediate_states=True): A tuple (o, final_state, A_state, h)
            o (torch.Tensor):
                Outputs of shape `[B, T, H, V]`.
            final_state (torch.Tensor):
                Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
            A_state (torch.Tensor):
                Final ATK diagonal state of shape `[N, H, K]` if `output_final_state=True` else `None`.
            h (torch.Tensor):
                Intermediate states of shape `[B, NT, H, K, V]` and dtype `bfloat16` for caching or further processing.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.precond_kda import chunk_precond_kda
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')
        >>> beta_atk = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')
        >>> g = torch.rand(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> g_atk = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> A_log = torch.randn(H, dtype=torch.float32, device='cuda')
        >>> dt_bias = torch.randn(H * K, dtype=torch.float32, device='cuda')
        >>> o, ht, at = chunk_precond_kda(
            q, k, v, g, g_atk, beta_atk, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_gate_in_kernel=True,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, beta_atk, g, g_atk = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, beta_atk, g, g_atk))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht, at = chunk_precond_kda(
            q, k, v, g, g_atk, beta_atk, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_gate_in_kernel=True,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """

    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        # Override cu_seqlens and cu_seqlens_cpu with the ones from the context
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
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert k.shape[-1] <= 256, "Currently we only support key headdim <=256 for KDA :-("
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num of head)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Prepare log_atk_scale with default if needed
    H = k.shape[2]
    if log_atk_scale is None:
        log_atk_scale = torch.full((H,), -0.2, device=k.device, dtype=torch.float32)

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    results = ChunkPrecondKDAFunction.apply(
        q,
        k,
        v,
        g,
        g_atk,
        beta_atk,
        beta,
        scale,
        initial_state,
        initial_A_state,
        output_final_state,
        True,   # use_qk_l2norm_in_kernel (always True)
        use_gate_in_kernel,
        A_log,
        dt_bias,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
        safe_gate,
        cp_context,
        transpose_state_layout,
        x,
        eps,
        log_atk_scale,
        lower_bound,
        solve_tril_precision,
        disable_recompute,
        return_intermediate_states,
    )

    if return_intermediate_states:
        # returns (o, final_state, at, h)
        return results

    o, h_final, a_final = results
    return o, h_final, a_final
