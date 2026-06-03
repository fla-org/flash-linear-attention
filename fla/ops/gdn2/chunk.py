# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

r"""
Chunkwise training kernel for GDN-2 (Gated DeltaNet 2).

GDN-2 extends the gated delta rule with two independent channel-wise gates: an
erase gate ``b in R^K`` on the key axis and a write gate ``w in R^V`` on the
value axis. The per-token recurrence on the matrix state ``S in R^{K x V}`` is

    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T

where ``*`` is the elementwise product. Setting ``b_t = w_t = beta`` (scalar)
recovers KDA; further collapsing ``g_t`` to a scalar recovers Gated DeltaNet v1.

This file is the public entry point and the autograd wrapper. Forward and
backward orchestration live in ``chunk_fwd.py`` and ``chunk_bwd.py``; the
GDN-2-specific Triton kernels live in ``chunk_intra.py``,
``chunk_intra_token_parallel.py``, ``wy_fast.py``, and ``chunk_bwd.py``.
"""

from __future__ import annotations

import warnings

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.gdn2.chunk_bwd import chunk_gdn2_bwd
from fla.ops.gdn2.chunk_fwd import chunk_gdn2_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class ChunkGDN2Function(torch.autograd.Function):
    """Autograd-compatible wrapper around GDN-2 forward / backward."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        A_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        use_gate_in_kernel: bool,
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_cpu: torch.LongTensor | None,
        safe_gate: bool,
        lower_bound: float | None,
        chunk_size: int,
        disable_recompute: bool,
        return_intermediate_states: bool,
        state_v_first: bool,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, chunk_size)
            if cu_seqlens is not None else None
        )

        g_input = g

        (o, final_state, g_cumsum, Aqk, Akk,
         w_wy, u_wy, qg, kg, v_new, h, initial_state) = chunk_gdn2_fwd(
            q=q,
            k=k,
            v=v,
            g=g_input,
            b=b,
            w_gate=w,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            state_v_first=state_v_first,
        )

        if return_intermediate_states:
            assert torch.is_inference_mode_enabled(), (
                "return_intermediate_states is only allowed in inference mode"
            )
            assert disable_recompute is False, (
                "return_intermediate_states must be used with disable_recompute=False"
            )
            return o.type_as(q), final_state, h

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g_cumsum, g_input, b, w, A_log, dt_bias,
            Aqk, Akk, w_wy, u_wy, qg, kg, v_new, h,
            initial_state, cu_seqlens, chunk_indices,
        )
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.disable_recompute = disable_recompute
        ctx.state_v_first = state_v_first
        return o.type_as(q), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        (q, q_rstd, k, k_rstd, v, g_cumsum, g_input, b, w, A_log, dt_bias,
         Aqk, Akk, w_wy, u_wy, qg, kg, v_new, h,
         initial_state, cu_seqlens, chunk_indices) = ctx.saved_tensors

        dq, dk, dv, db, dw, dg, dh0, dA_log, dt_bias_grad = chunk_gdn2_bwd(
            q=q,
            k=k,
            v=v,
            b=b,
            w_gate=w,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            g=g_cumsum,
            g_org=g_input if ctx.use_gate_in_kernel else None,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            state_v_first=ctx.state_v_first,
            w_wy=w_wy, u_wy=u_wy, qg=qg, kg=kg, v_new=v_new, h=h,
            disable_recompute=ctx.disable_recompute,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return (
            dq.to(q.dtype),          # q
            dk.to(k.dtype),          # k
            dv.to(v.dtype),          # v
            dg.to(g_input.dtype),    # g
            db.to(b.dtype),          # b
            dw.to(w.dtype),          # w
            dA_log,                  # A_log
            dt_bias_grad,            # dt_bias
            None,                    # scale
            dh0,                     # initial_state
            None,                    # output_final_state
            None,                    # use_qk_l2norm_in_kernel
            None,                    # use_gate_in_kernel
            None,                    # cu_seqlens
            None,                    # cu_seqlens_cpu
            None,                    # safe_gate
            None,                    # lower_bound
            None,                    # chunk_size
            None,                    # disable_recompute
            None,                    # return_intermediate_states
            None,                    # state_v_first
        )


@torch.compiler.disable
def chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    **kwargs,
):
    r"""
    Chunkwise forward for Gated DeltaNet 2 (GDN-2).

    The matrix-state recurrence on ``S in R^{K x V}`` is

        S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T

    where ``b in R^K`` and ``w in R^V`` are channel-wise erase and write gates.
    Setting ``b = w = beta`` (scalar) recovers KDA.

    Args:
        q: queries of shape ``[B, T, H, K]``.
        k: keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        g: (forget) log-decay of shape ``[B, T, H, K]``. With
            ``use_gate_in_kernel=True`` this is the raw pre-activation and the
            kernel computes ``-exp(A_log) * softplus(g + dt_bias)`` (or the
            bounded variant if ``lower_bound`` is set).
        b: channel-wise ERASE gate of shape ``[B, T, H, K]``. Replaces KDA's
            scalar beta. Typical range: ``[0, 2]``.
        w: channel-wise WRITE gate of shape ``[B, T, H, V]``. New for GDN-2.
            Typical range: ``[0, 1]``.
        scale: attention scale. Defaults to ``1 / sqrt(K)``.
        initial_state: optional ``[N, H, K, V]`` initial state in float32 (or
            ``[N, H, V, K]`` if ``state_v_first=True``).
        output_final_state: whether to output the final recurrent state.
        use_qk_l2norm_in_kernel: L2-normalize q and k inside the kernel.
        use_gate_in_kernel: compute the gate activation inside the kernel from
            raw ``g`` and (required) ``A_log`` (plus optional ``dt_bias``,
            ``lower_bound``).
        cu_seqlens: ``[N+1]`` packed-sequence offsets. Requires batch size 1.
        cu_seqlens_cpu: optional CPU mirror of ``cu_seqlens``, forwarded to the
            state-recurrence kernel.
        safe_gate: use the safe-gate intra kernel variant (M=16 TensorCore
            path; requires gate values in ``[-5, 0)`` if combined with
            ``use_gate_in_kernel=True``).
        lower_bound: when ``safe_gate=True`` and ``use_gate_in_kernel=True``,
            use the bounded gate activation
            ``lower_bound * sigmoid(exp(A_log) * g)``. Must lie in ``[-5, 0)``.
        disable_recompute: retain forward intermediates for a faster backward
            at the cost of memory. Default: ``False``.
        return_intermediate_states: when ``True``, also returns the per-chunk
            pre-states ``h`` (shape ``[B, NT, H, K, V]``). Must be used inside
            ``torch.inference_mode()``.
        state_v_first: store the recurrent state in ``[V, K]`` layout instead
            of the default ``[K, V]``.

    Returns:
        - Normal mode: ``(o, final_state)``.
        - Intermediate mode (``return_intermediate_states=True``):
          ``(o, final_state, h)``.

    Examples::

        >>> import torch
        >>> import torch.nn.functional as F
        >>> from fla.ops.gdn2 import chunk_gdn2
        >>> B, T, H, K, V = 2, 1024, 4, 128, 128
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, H, K, device='cuda')).to(torch.bfloat16)
        >>> b = torch.rand(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> w = torch.rand(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gdn2(q, k, v, g, b, w, output_final_state=True)
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

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    chunk_size = kwargs.pop("chunk_size", 64)
    if chunk_size != 64:
        # The blocked solver in chunk_intra is hardcoded to NC=4 sub-chunks of
        # size 16, which assumes BT == 64.
        raise ValueError(f"`chunk_size` must be 64 for GDN-2, got {chunk_size}.")

    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert k.shape[-1] <= 256, f"GDN-2 only supports key headdim <= 256, got {k.shape[-1]}."
    assert b.shape == q.shape, (
        f"b (channel-wise erase gate) must have shape [B, T, H, K] matching q; "
        f"got {tuple(b.shape)} vs q {tuple(q.shape)}."
    )
    assert w.shape == v.shape, (
        f"w (channel-wise write gate) must have shape [B, T, H, V] matching v; "
        f"got {tuple(w.shape)} vs v {tuple(v.shape)}."
    )
    assert v.shape == (*q.shape[:3], v.shape[-1]), (
        f"v must have shape [B, T, H, V] matching q on (B, T, H); "
        f"got {tuple(v.shape)} vs q {tuple(q.shape)}."
    )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    return ChunkGDN2Function.apply(
        q,
        k,
        v,
        g,
        b,
        w,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        safe_gate,
        lower_bound,
        chunk_size,
        disable_recompute,
        return_intermediate_states,
        state_v_first,
    )
