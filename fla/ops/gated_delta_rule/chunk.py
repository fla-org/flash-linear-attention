# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.backends import dispatch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.common.gate import fused_beta_sigmoid, fused_beta_sigmoid_bwd
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    chunk_gated_delta_rule_fwd_h_pre_process,
    compress_h0,
    expand_h0,
)
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.gate import gdn_gate_bwd, gdn_gate_chunk_cumsum
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.graph import (
    host_chunk_statistics,
    is_graph_capable_device,
    normalize_graph_mode,
    route_graph_execution,
    static_chunk_capacity,
)
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_indices_static
from fla.utils import IS_NPU, IS_NVIDIA, autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    use_graph: bool = False,
):
    graph_kwargs = {'use_graph': True} if use_graph else {}
    g_input = g if use_gate_in_kernel else None
    if use_gate_in_kernel:
        g = gdn_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            **graph_kwargs,
        )
    else:
        g = chunk_local_cumsum(
            g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            **graph_kwargs,
        )
    # obtain WY representation. u is actually the new v.
    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        **graph_kwargs,
    )

    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=k,
            w=w,
            u=u,
            g=g,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            context=cp_context,
            state_v_first=state_v_first,
            chunk_size=chunk_size,
        )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
        **({'chunk_offsets': chunk_offsets, 'use_graph': True} if use_graph else {}),
    )

    if cp_context is not None:
        initial_state = compress_h0(initial_state, context=cp_context)

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
        **graph_kwargs,
    )
    return g, o, A, final_state, initial_state, g_input


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False,
    g_input: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    use_graph: bool = False,
):
    graph_kwargs = {'use_graph': True} if use_graph else {}
    graph_state_kwargs = {'chunk_offsets': chunk_offsets, 'use_graph': True} if use_graph else {}
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        **graph_kwargs,
    )

    if cp_context is not None:
        initial_state = expand_h0(initial_state, context=cp_context)

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
        **graph_state_kwargs,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        **graph_kwargs,
    )

    if cp_context is not None:
        # initial_state is None in the CP mode
        # We only need to compute dht of current rank and pass it to the backward kernel
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            dht=dht,
            initial_state=initial_state,
            context=cp_context,
            state_v_first=state_v_first,
            chunk_size=chunk_size,
        )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
        **graph_state_kwargs,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
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
        state_v_first=state_v_first,
        chunk_size=chunk_size,
        **graph_kwargs,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        **graph_kwargs,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        **graph_kwargs,
    )
    dA_log, ddt_bias = None, None
    if use_gate_in_kernel:
        gate_kwargs = {'cu_seqlens': cu_seqlens, 'use_graph': True} if use_graph else {}
        dg, dA_log, ddt_bias = gdn_gate_bwd(g=g_input, A_log=A_log, dt_bias=dt_bias, dyg=dg, **gate_kwargs)
    return dq, dk, dv, db, dg, dh0, dA_log, ddt_bias


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

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
        initial_state: torch.Tensor,
        output_final_state: bool,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        cp_context: FLACPContext | None = None,
        use_graph: bool = False,
        chunk_size: int = 64,
        chunk_offsets: torch.LongTensor | None = None,
        graph_nt_max: int | None = None,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        beta_raw = beta
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0)

        if use_graph:
            if graph_nt_max is None:
                n_max = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else 1
                graph_nt_max = static_chunk_capacity(q.shape[1], n_max, chunk_size)
            if cu_seqlens is not None:
                if chunk_indices is None:
                    chunk_indices, generated_offsets = prepare_chunk_indices_static(cu_seqlens, chunk_size, graph_nt_max)
                    if chunk_offsets is None:
                        chunk_offsets = generated_offsets
                elif chunk_offsets is None:
                    _, chunk_offsets = prepare_chunk_indices_static(cu_seqlens, chunk_size, graph_nt_max)
        elif chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        g, o, A, final_state, initial_state, g_input = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            state_v_first=state_v_first,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=chunk_size,
            use_graph=use_graph,
        )
        ctx.save_for_backward(
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g,
            beta_raw,
            beta,
            A,
            initial_state,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            g_input,
            A_log,
            dt_bias,
        )
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.allow_neg_eigval = allow_neg_eigval
        ctx.cp_context = cp_context
        ctx.state_v_first = state_v_first
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_graph = use_graph
        ctx.nt_max = graph_nt_max if use_graph else None
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
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g,
            beta_raw,
            beta,
            A,
            initial_state,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            g_input,
            A_log,
            dt_bias,
        ) = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0, dA_log, ddt_bias = chunk_gated_delta_rule_bwd(
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
            cu_seqlens=cu_seqlens,
            cp_context=ctx.cp_context,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            state_v_first=ctx.state_v_first,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            g_input=g_input,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=ctx.chunk_size,
            use_graph=ctx.use_graph,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        if ctx.use_beta_sigmoid_in_kernel:
            db = fused_beta_sigmoid_bwd(beta_raw, db, scale=2.0 if ctx.allow_neg_eigval else 1.0)
        return (
            dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta_raw),
            None, dh0, None, None, None, None, None, None, None, dA_log, ddt_bias,
            None, None, None, None, None, None, None,
        )


@torch.compiler.disable
@dispatch('gated_delta_rule')
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    use_graph: bool = False,
    graph_mode: str | None = None,
    graph_t_max: int | None = None,
    graph_n_max: int | None = None,
    graph_nt_max: int | None = None,
    graph_actual_tokens: int | None = None,
    graph_actual_sequences: int | None = None,
    graph_actual_nt: int | None = None,
    min_graph_utilization: float = 0.75,
    chunk_offsets: torch.LongTensor | None = None,
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
        g (torch.Tensor):
            (forget) gating tensor of shape `[B, T, HV]`.
            When `use_gate_in_kernel=False` (default), `g` should be in log space (pre-computed decay).
            When `use_gate_in_kernel=True`, `g` is the raw input before gate activation;
            the kernel fuses `-exp(A_log) * softplus(g + dt_bias)` + chunk cumsum internally.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space GDN decay internally.
            When `True`, the passed `g` is the raw input, and `A_log` must be provided.
            The kernel fuses gate activation + chunk cumsum in a single pass.
            Default: `False`.
        A_log (Optional[torch.Tensor]):
            Decay parameter of shape `[HV]`. Required when `use_gate_in_kernel=True`.
        dt_bias (Optional[torch.Tensor]):
            Bias added to `g` before activation, of shape `[HV]`.
            Only used when `use_gate_in_kernel=True`.
        use_beta_sigmoid_in_kernel (bool):
            Whether to apply `torch.sigmoid(beta)` before launching the chunk kernel.
            - If `True`, the passed `beta` acts as the raw beta logits.
            - If `False`, `beta` is expected to already be in post-sigmoid space.
            Default: `False`.
        allow_neg_eigval (bool):
            Whether to allow negative eigenvalues by scaling `beta` to `[0, 2)`.
            Only takes effect together with `use_beta_sigmoid_in_kernel=True`, in which case
            the kernel computes `2 * sigmoid(beta)` instead of `sigmoid(beta)`. Default: `False`.
        state_v_first (Optional[bool]):
            Store the recurrent state in V-first ``[V, K]`` layout instead of the default ``[K, V]``. Default: ``False``.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        chunk_indices (Optional[torch.LongTensor]):
            Pre-computed chunk indices for variable-length inputs.
            If provided, they are used directly instead of being computed from `cu_seqlens`. Default: `None`.
        cp_context (Optional[FLACPContext]):
            Context parallel context for distributed training across multiple devices.
            When provided, `initial_state` and `output_final_state` are not supported,
            and `cu_seqlens` will be overridden by the context. Default: `None`.
        use_graph (Optional[bool]):
            Whether to use fixed-capacity chunk metadata for CUDA Graph capture and replay. This legacy compatibility
            switch may build static metadata inside the call; use explicit `graph_mode="force_graph"` when the caller
            owns fixed-address metadata. Default: `False`.
        graph_mode (Optional[str]):
            Graph strategy: `"eager"`, `"force_graph"`, or `"auto"`. `None` preserves the
            `use_graph` behavior. Routing happens outside graph capture. Default: `None`.
        graph_t_max (Optional[int]):
            Maximum token capacity of the graph bucket. Defaults to `q.shape[1]`.
        graph_n_max (Optional[int]):
            Maximum sequence capacity of the graph bucket. Defaults to `len(cu_seqlens) - 1`.
        graph_nt_max (Optional[int]):
            Maximum chunk-slot capacity. Defaults to `ceil(graph_t_max / chunk_size) + graph_n_max - 1`.
        graph_actual_tokens (Optional[int]):
            Host-side actual token count for `graph_mode="auto"`. Default: `None`.
        graph_actual_sequences (Optional[int]):
            Host-side non-empty sequence count for `graph_mode="auto"`. Default: `None`.
        graph_actual_nt (Optional[int]):
            Host-side actual chunk count for `graph_mode="auto"`. Default: `None`.
        min_graph_utilization (float):
            Minimum `actual_nt / graph_nt_max` required by `"auto"`. Default: `0.75`.
        chunk_offsets (Optional[torch.LongTensor]):
            Fixed-shape chunk prefix offsets for an externally managed graph bucket. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, HV, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, HV, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if 'transpose_state_layout' in kwargs:
        if state_v_first:
            raise ValueError("Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`.")
        warnings.warn(
            "`transpose_state_layout` is deprecated and renamed to `state_v_first`.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop('transpose_state_layout')

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
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )

    chunk_size = kwargs.pop('chunk_size', 64)
    if chunk_size not in (16, 32, 64):
        raise ValueError(f"`chunk_size` must be 16, 32, or 64 for Gated Delta Rule, got {chunk_size}.")

    normalized_graph_mode = normalize_graph_mode(graph_mode, use_graph)
    if normalized_graph_mode != "eager":
        if IS_NPU:
            if normalized_graph_mode == 'force_graph':
                raise NotImplementedError("GDN graph mode is not implemented for the Ascend backend yet.")
            graph_mode = 'eager'
        elif not IS_NVIDIA:
            if normalized_graph_mode == 'force_graph':
                raise RuntimeError("Graph mode requires the CUDA or Ascend NPU backend.")
            graph_mode = 'eager'
        if cp_context is not None:
            if normalized_graph_mode == 'force_graph':
                raise ValueError("CUDA Graph mode does not support Context Parallel.")
            graph_mode = 'eager'
        if cu_seqlens_cpu is not None and cu_seqlens_cpu.device.type != 'cpu':
            raise ValueError("graph `cu_seqlens_cpu` must be a CPU tensor")
        if cu_seqlens is None and q.shape[0] != 1:
            if normalized_graph_mode == "force_graph":
                raise ValueError("Dense graph mode requires batch size 1 (the N=1 special case).")
            graph_mode = 'eager'
        if graph_mode == 'eager':
            use_graph = False

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
    use_gate_in_kernel = kwargs.get('use_gate_in_kernel', False)
    A_log = kwargs.get('A_log')
    dt_bias = kwargs.get('dt_bias')
    if use_gate_in_kernel:
        assert A_log is not None, "A_log must be provided when use_gate_in_kernel=True."
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    if scale is None:
        scale = k.shape[-1] ** -0.5
    if graph_t_max is None:
        graph_t_max = q.shape[1]
    if graph_n_max is None:
        graph_n_max = len(cu_seqlens) - 1 if cu_seqlens is not None else 1
    if graph_nt_max is None and chunk_indices is not None and normalized_graph_mode != "eager":
        if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
            raise ValueError("graph chunk_indices must have shape [NT_max, 2]")
        graph_nt_max = chunk_indices.shape[0]
    if graph_nt_max is None:
        graph_nt_max = static_chunk_capacity(graph_t_max, graph_n_max, chunk_size)

    actual_tokens, actual_sequences, actual_nt = graph_actual_tokens, graph_actual_sequences, graph_actual_nt
    if cu_seqlens is None:
        actual_tokens = q.shape[1] if actual_tokens is None else actual_tokens
        actual_sequences = 1 if actual_sequences is None else actual_sequences
        actual_nt = (q.shape[1] + chunk_size - 1) // chunk_size if actual_nt is None else actual_nt
    if cu_seqlens_cpu is not None and any(value is None for value in (actual_tokens, actual_sequences, actual_nt)):
        host_tokens, host_sequences, host_nt = host_chunk_statistics(cu_seqlens_cpu, chunk_size)
        actual_tokens = host_tokens if actual_tokens is None else actual_tokens
        actual_sequences = host_sequences if actual_sequences is None else actual_sequences
        actual_nt = host_nt if actual_nt is None else actual_nt
    elif cu_seqlens is not None and cu_seqlens.device.type == 'cpu' and any(
        value is None for value in (actual_tokens, actual_sequences, actual_nt)
    ):
        host_tokens, host_sequences, host_nt = host_chunk_statistics(cu_seqlens, chunk_size)
        actual_tokens = host_tokens if actual_tokens is None else actual_tokens
        actual_sequences = host_sequences if actual_sequences is None else actual_sequences
        actual_nt = host_nt if actual_nt is None else actual_nt

    decision = route_graph_execution(
        graph_mode,
        use_graph=use_graph,
        actual_tokens=actual_tokens,
        actual_sequences=actual_sequences,
        actual_nt=actual_nt,
        t_max=graph_t_max,
        n_max=graph_n_max,
        nt_max=graph_nt_max,
        min_graph_utilization=min_graph_utilization,
        chunk_size=chunk_size,
        input_tokens=q.shape[1],
        input_sequences=1 if cu_seqlens is None else len(cu_seqlens) - 1,
    )
    use_graph = decision.selected_path == "graph"
    if use_graph:
        if not (IS_NVIDIA or IS_NPU):
            raise RuntimeError("Graph mode requires the CUDA or Ascend NPU backend.")
        if cu_seqlens is None:
            if q.shape[0] != 1:
                raise ValueError("Dense graph mode requires batch size 1 (the N=1 special case).")
            if graph_n_max != 1:
                raise ValueError(f"dense graph mode requires graph_n_max=1, got {graph_n_max}")
        elif not is_graph_capable_device(cu_seqlens.device):
            raise ValueError("Graph mode requires device-resident `cu_seqlens`.")
        if q.shape[1] != graph_t_max:
            raise ValueError(
                f"graph input shape must use graph_t_max={graph_t_max}, got q.shape[1]={q.shape[1]}"
            )
        if cu_seqlens is not None and graph_n_max != len(cu_seqlens) - 1:
            raise ValueError(
                f"graph_n_max={graph_n_max} must equal the fixed cu_seqlens capacity {len(cu_seqlens) - 1}"
            )
        if cu_seqlens is not None and graph_mode == "force_graph" and (chunk_indices is None or chunk_offsets is None):
            raise ValueError(
                "graph_mode='force_graph' requires caller-provided fixed `chunk_indices` and `chunk_offsets`"
            )
        if chunk_indices is not None:
            if chunk_indices.shape != (graph_nt_max, 2):
                raise ValueError(
                    f"graph chunk_indices must have shape {(graph_nt_max, 2)}, got {tuple(chunk_indices.shape)}"
                )
            metadata_device = cu_seqlens.device if cu_seqlens is not None else q.device
            metadata_dtype = cu_seqlens.dtype if cu_seqlens is not None else torch.long
            if chunk_indices.device != metadata_device or chunk_indices.dtype != metadata_dtype:
                raise ValueError("graph chunk_indices must share device and dtype with the graph metadata")
            if not chunk_indices.is_contiguous():
                raise ValueError("graph chunk_indices must be contiguous")
        if chunk_offsets is not None:
            if chunk_offsets.shape != (graph_n_max + 1,):
                raise ValueError(
                    f"graph chunk_offsets must have shape {(graph_n_max + 1,)}, got {tuple(chunk_offsets.shape)}"
                )
            metadata_device = cu_seqlens.device if cu_seqlens is not None else q.device
            metadata_dtype = cu_seqlens.dtype if cu_seqlens is not None else torch.long
            if chunk_offsets.device != metadata_device or chunk_offsets.dtype != metadata_dtype:
                raise ValueError("graph chunk_offsets must share device and dtype with the graph metadata")
            if not chunk_offsets.is_contiguous():
                raise ValueError("graph chunk_offsets must be contiguous")
    elif normalized_graph_mode == "auto" or graph_mode == "eager":
        # Static graph metadata describes the bucket, not an eager request.
        # Rebuild dynamic metadata in the eager path instead of exposing
        # sentinel rows to kernels that do not enable graph guards.
        chunk_indices = None
        chunk_offsets = None
    if normalized_graph_mode == 'auto' and not use_graph:
        # Re-enter once with eager so existing backend priority still applies.
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            graph_mode='eager',
            chunk_size=chunk_size,
            **kwargs,
        )
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        cp_context,
        use_graph,
        chunk_size,
        chunk_offsets,
        graph_nt_max,
    )
    return o, final_state


chunk_gdn = chunk_gated_delta_rule
