# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch

from fla.modules.norm.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.gate import fused_beta_sigmoid, fused_beta_sigmoid_bwd
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from fla.ops.triadic_kda.chunk_bwd import chunk_triadic_kda_bwd
from fla.ops.triadic_kda.chunk_fwd import CHUNK_SIZE, chunk_triadic_kda_fwd
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class ChunkTriadicKDAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k2: torch.Tensor,
        q2: torch.Tensor,
        gk: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        A2_log: torch.Tensor | None,
        dt2_bias: torch.Tensor | None,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        use_gate_in_kernel: bool,
        use_beta_sigmoid_in_kernel: bool,
        allow_neg_eigval: bool,
        lower_bound: float | None,
        cu_seqlens: torch.Tensor | None,
        disable_recompute: bool,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        beta_raw = beta
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(x=beta_raw, scale=2.0 if allow_neg_eigval else 1.0)
        beta_f = beta.float()

        indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
        gk_input, g2_input = gk, g2
        if use_gate_in_kernel:
            gk_cum = kda_gate_chunk_cumsum(
                g=gk_input,
                A_log=A_log,
                chunk_size=CHUNK_SIZE,
                dt_bias=dt_bias,
                cu_seqlens=cu_seqlens,
                chunk_indices=indices,
                lower_bound=lower_bound,
            )
            g2_cum = kda_gate_chunk_cumsum(
                g=g2_input,
                A_log=A2_log,
                chunk_size=CHUNK_SIZE,
                dt_bias=dt2_bias,
                cu_seqlens=cu_seqlens,
                chunk_indices=indices,
                lower_bound=lower_bound,
            )
        else:
            gk_cum = chunk_local_cumsum(gk_input.float(), chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=indices)
            g2_cum = chunk_local_cumsum(g2_input.float(), chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=indices)

        o, ht, saved = chunk_triadic_kda_fwd(
            q=q,
            k=k,
            v=v,
            k2=k2.float(),
            q2=q2.float(),
            gk=gk_cum,
            g2=g2_cum,
            beta=beta_f,
            h0=initial_state,
            cu=cu_seqlens,
            scale=scale,
            save=disable_recompute,
            final=output_final_state,
            indices=indices,
        )
        ctx.save_for_backward(*saved, q_rstd, k_rstd, gk_input, g2_input, beta_raw, A_log, dt_bias, A2_log, dt2_bias)
        ctx.scale = scale
        ctx.k2_dtype, ctx.q2_dtype = k2.dtype, q2.dtype
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.allow_neg_eigval = allow_neg_eigval
        ctx.lower_bound = lower_bound
        ctx.set_materialize_grads(False)
        return o.to(v.dtype), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        *saved, q_rstd, k_rstd, gk_input, g2_input, beta_raw, A_log, dt_bias, A2_log, dt2_bias = ctx.saved_tensors
        v, beta_f, q, k, q2, k2, gk_cum, g2_cum, Ai, Aqk, states, u, Lk, Mk, pq, pk, pr, h0, cu, indices = saved
        if do is None:
            do = torch.zeros_like(v)

        dq, dk, dv, dk2, dq2, dgk, dg2, dbeta, dh0 = chunk_triadic_kda_bwd(
            v=v,
            beta=beta_f,
            q=q,
            k=k,
            q2=q2,
            k2=k2,
            gk=gk_cum,
            g2=g2_cum,
            Ai=Ai,
            Aqk=Aqk,
            states=states,
            u=u,
            Lk=Lk,
            Mk=Mk,
            h0=h0,
            cu=cu,
            indices=indices,
            do=do,
            dht=dht,
            scale=ctx.scale,
            keys=[pq, pk, pr],
        )

        dA, dbias, dA2, dbias2 = None, None, None, None
        if ctx.use_gate_in_kernel:
            dgk, dA, dbias = kda_gate_bwd(g=gk_input, A_log=A_log, dt_bias=dt_bias, dyg=dgk, lower_bound=ctx.lower_bound)
            dg2, dA2, dbias2 = kda_gate_bwd(g=g2_input, A_log=A2_log, dt_bias=dt2_bias, dyg=dg2, lower_bound=ctx.lower_bound)
        else:
            dgk, dg2 = dgk.to(gk_input), dg2.to(g2_input)

        if ctx.use_beta_sigmoid_in_kernel:
            dbeta = fused_beta_sigmoid_bwd(x=beta_raw, dy=dbeta, scale=2.0 if ctx.allow_neg_eigval else 1.0)
        else:
            dbeta = dbeta.to(beta_raw)

        if ctx.use_qk_l2norm_in_kernel:
            dq, dk = l2norm_bwd(q, q_rstd, dq), l2norm_bwd(k, k_rstd, dk)
        else:
            dq, dk = dq.to(q), dk.to(k)

        return (
            dq,
            dk,
            dv.to(v),
            dk2.to(ctx.k2_dtype),
            dq2.to(ctx.q2_dtype),
            dgk,
            dg2,
            dbeta,
            dA,
            dbias,
            dA2,
            dbias2,
            None,
            dh0.to(h0) if dh0 is not None else None,
            *([None] * 8),
        )


def _validate_inputs(
    q,
    k,
    v,
    k2,
    q2,
    gk,
    g2,
    beta,
    initial_state,
    cu_seqlens,
    scale,
    use_gate_in_kernel,
    A_log,
    dt_bias,
    A2_log,
    dt2_bias,
    use_beta_sigmoid_in_kernel,
    allow_neg_eigval,
    lower_bound,
):
    if k.ndim != 4:
        raise ValueError('chunk_triadic_kda expects [B, T, H, D] inputs.')
    B, T, H, K = k.shape
    E, V = k2.shape[-1], v.shape[-1]
    if min(B, T, H, K, V, E) < 1 or K > 256 or V > 256:
        raise ValueError(
            f'chunk_triadic_kda supports K, V in [1, 256] and E >= 1, got B={B}, T={T}, H={H}, K={K}, V={V}, E={E}.',
        )
    if q.shape != k.shape or gk.shape != k.shape:
        raise ValueError(f'q, k and gk must share shape [B, T, H, K]={[B, T, H, K]}.')
    if v.shape != (B, T, H, V):
        raise ValueError(f'v must have shape [B, T, H, V]={[B, T, H, V]}, got {list(v.shape)}.')
    if any(x.shape != (B, T, H, E) for x in (k2, q2, g2)):
        raise ValueError(f'k2, q2 and g2 must share shape [B, T, H, E]={[B, T, H, E]}.')
    if beta.shape != (B, T, H):
        raise ValueError(f'beta must have shape [B, T, H]={[B, T, H]}, got {list(beta.shape)}.')

    N = B
    if cu_seqlens is not None:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError('Packed inputs require B=1 and 1-D int32/int64 cu_seqlens.')
        N = len(cu_seqlens) - 1
    if initial_state is not None and initial_state.shape != (N, H, E, K, V):
        raise ValueError(
            f'initial_state must have shape [N, H, E, K, V]={[N, H, E, K, V]}, got {list(initial_state.shape)}.',
        )
    if scale is not None and not math.isfinite(scale):
        raise ValueError('scale must be finite.')
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError('`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.')
    if use_gate_in_kernel:
        if lower_bound is not None and (not math.isfinite(lower_bound) or lower_bound >= 0):
            raise ValueError('`lower_bound` must be finite and negative when specified.')
        if lower_bound is None and (A_log is None or A2_log is None):
            raise ValueError('`A_log` and `A2_log` are required when `use_gate_in_kernel=True` without `lower_bound`.')
        for name, param, expected in (
            ('A_log', A_log, H),
            ('A2_log', A2_log, H),
            ('dt_bias', dt_bias, H * K),
            ('dt2_bias', dt2_bias, H * E),
        ):
            if param is not None and param.numel() != expected:
                raise ValueError(f'`{name}` must have {expected} elements, got {param.numel()}.')


@torch.compiler.disable
def chunk_triadic_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    gk: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    A2_log: torch.Tensor | None = None,
    dt2_bias: torch.Tensor | None = None,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    disable_recompute: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Chunkwise-parallel Triadic Kimi Delta Attention with dual-axis decay (`gk` on `K` and `g2` on `E`)
    and scalar write strength `beta`:

        S_{e,k} <- exp(g2_{t,e} + gk_{t,k}) * S_{e,k}
        S       <- S + beta_t * (k2_t ⊗ k_t) (v_t - (k2_t ⊗ k_t)^T S)^T
        o_t     = scale * S^T (q2_t ⊗ q_t)

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        k2 (torch.Tensor):
            Second keys of shape `[B, T, H, E]`.
        q2 (torch.Tensor):
            Second queries of shape `[B, T, H, E]`.
        gk (torch.Tensor):
            First-axis decays of shape `[B, T, H, K]`. Log-space when `use_gate_in_kernel=False`,
            or raw pre-activation logits when `use_gate_in_kernel=True`.
        g2 (torch.Tensor):
            Second-axis decays of shape `[B, T, H, E]`. Log-space when `use_gate_in_kernel=False`,
            or raw pre-activation logits when `use_gate_in_kernel=True`.
        beta (torch.Tensor, Optional):
            Scalar write strengths of shape `[B, T, H]`. Default: `None` (ones).
        scale (float, Optional):
            Output scale factor. Default: `None` (`K ** -0.5`).
        initial_state (torch.Tensor, Optional):
            Initial state of shape `[N, H, E, K, V]`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the FP32 final state of shape `[N, H, E, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool, Optional):
            Whether to fuse L2 normalization on `q` and `k`. Default: `False`.
        use_gate_in_kernel (bool, Optional):
            Whether to fuse `-exp(A_log) * softplus(gk + dt_bias)` and `-exp(A2_log) * softplus(g2 + dt2_bias)`
            (or `lower_bound * sigmoid(...)` when `lower_bound` is set) with intra-chunk cumsum. Default: `False`.
        A_log (torch.Tensor, Optional):
            First-axis log decay scale of shape `[H]`. Default: `None`.
        dt_bias (torch.Tensor, Optional):
            First-axis decay bias of shape `[H * K]`. Default: `None`.
        A2_log (torch.Tensor, Optional):
            Second-axis log decay scale of shape `[H]`. Default: `None`.
        dt2_bias (torch.Tensor, Optional):
            Second-axis decay bias of shape `[H * E]`. Default: `None`.
        use_beta_sigmoid_in_kernel (bool, Optional):
            Whether to fuse `sigmoid(beta)` (scaled by `2` if `allow_neg_eigval=True`). Default: `False`.
        allow_neg_eigval (bool, Optional):
            Whether to allow negative eigenvalues (`beta in (0, 2)`). Default: `False`.
        lower_bound (float, Optional):
            Optional negative lower bound in log space when `use_gate_in_kernel=True`. Default: `None`.
        cu_seqlens (torch.Tensor, Optional):
            Cumulative sequence lengths of shape `[N + 1]` (int32 or int64) for variable-length inputs (`B == 1`).
            Default: `None`.
        disable_recompute (bool, Optional):
            Whether to cache chunk states and solved residuals for backward. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            FP32 final state of shape `[N, H, E, K, V]` if `output_final_state=True`, else `None`.
    """
    B, T, H, K = k.shape
    scale = K ** -0.5 if scale is None else scale
    if beta is None:
        beta = torch.ones(B, T, H, device=q.device, dtype=torch.float32)
    if not use_gate_in_kernel:
        A_log = dt_bias = A2_log = dt2_bias = lower_bound = None
    _validate_inputs(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        A2_log=A2_log,
        dt2_bias=dt2_bias,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        lower_bound=lower_bound,
    )
    return ChunkTriadicKDAFunction.apply(
        q,
        k,
        v,
        k2,
        q2,
        gk,
        g2,
        beta,
        A_log,
        dt_bias,
        A2_log,
        dt2_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        lower_bound,
        cu_seqlens,
        disable_recompute,
    )
