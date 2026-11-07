# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.gka.chunk_solve_bwd import chunk_gka_solve_bwd
from fla.ops.gka.chunk_solve_fwd import chunk_gka_solve_fwd
from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class ChunkGKASolveFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        gk: torch.Tensor | None,
        ridge_ratio: float,
        num_iter: int,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_cpu: torch.LongTensor | None,
    ):
        chunk_size = 64
        chunk_indices = None
        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        gk_cs = None
        if gk is not None:
            gk_cs = chunk_local_cumsum(
                gk,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        h, ht = chunk_fwd_h(
            k=k,
            v=k,
            g=gk_cs,
            h0=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        x, fro = chunk_gka_solve_fwd(
            q=q,
            k=k,
            h=h,
            gk=gk_cs,
            ridge_ratio=ridge_ratio,
            num_iter=num_iter,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        ctx.save_for_backward(k, x, fro, gk_cs, initial_state, chunk_indices)
        ctx.gk_dtype = gk.dtype if gk is not None else None
        ctx.chunk_size = chunk_size
        ctx.ridge_ratio = ridge_ratio
        ctx.num_iter = num_iter
        ctx.cu_seqlens = cu_seqlens
        return x, ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dx, dht):
        k, x, fro, gk_cs, initial_state, chunk_indices = ctx.saved_tensors
        chunk_size, cu_seqlens = ctx.chunk_size, ctx.cu_seqlens
        # recompute the chunk-start states `H_{[c]}` instead of saving them, since they are K/64 times the size of `k`
        h, _ = chunk_fwd_h(
            k=k,
            v=k,
            g=gk_cs,
            h0=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        # the system matrix is symmetric, so the adjoint solve reuses the forward solve and its `fro`
        dq, _ = chunk_gka_solve_fwd(
            q=dx,
            k=k,
            h=h,
            gk=gk_cs,
            fro=fro,
            ridge_ratio=ctx.ridge_ratio,
            num_iter=ctx.num_iter,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        # d(A^{-1} q) = -A^{-1} dA x, so the gradient w.r.t. `H_t` is -sum(dq x^T); `chunk_bwd_dh` with `scale=1.`
        # accumulates +sum(dq x^T), hence the negated `dht` going in and `dh0` coming out
        dh, dh0 = chunk_bwd_dh(
            q=dq,
            k=k,
            v=k,
            do=x,
            h0=initial_state,
            dht=-dht if dht is not None else None,
            scale=1.,
            g=gk_cs,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        dk, dg, dh0_lamb = chunk_gka_solve_bwd(
            k=k,
            x=x,
            dq=dq,
            h=h,
            dh=dh,
            fro=fro,
            gk=gk_cs,
            ridge_ratio=ctx.ridge_ratio,
            output_dh0=initial_state is not None,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        dgk = None
        if dg is not None:
            dgk = chunk_local_cumsum(
                dg,
                chunk_size=chunk_size,
                reverse=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            ).to(ctx.gk_dtype)
        if dh0 is not None:
            dh0 = -dh0 + dh0_lamb
        return dq, dk, dgk, None, None, dh0, None, None, None


class SimpleGLABackwardGuard(torch.autograd.Function):
    """
    Identity in the forward; refuses the backward. Wraps `chunk_simple_gla`'s outputs for 16-bit inputs with small
    head dims, where its backward kernels hit illegal memory accesses (`chunk_bwd_kernel_dv` / `_dqkwg`, Triton 3.7.1).
    """

    @staticmethod
    def forward(ctx, o, h_kv, K, V):
        ctx.unsupported = (K, V, o.dtype)
        return o.view_as(o), (h_kv.view_as(h_kv) if h_kv is not None else None)

    @staticmethod
    def backward(ctx, do, dh_kv):
        K, V, dtype = ctx.unsupported
        raise RuntimeError(
            f"chunk_gka's backward is not supported for K={K}, V={V} in {dtype}: fla's chunk_simple_gla backward "
            "hits an illegal memory access for 16-bit inputs with K < 64 or V < 32. Use K >= 64 and V >= 32, or "
            "float32 inputs.",
        )


def check_gka_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    alpha: torch.Tensor | None,
    scale: float | None,
    num_iter: int,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float]:
    """
    Validates the inputs shared by `chunk_gka` and `fused_recurrent_gka`; returns `(h_kk0, h_kv0, scale)`.
    Raises `ValueError` on any unsupported shape or on `num_iter < 1`.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if k.shape != q.shape:
        raise ValueError(f"`k` must have the shape of `q` {tuple(q.shape)}, got {tuple(k.shape)}.")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(f"`v` must have shape [B, T, H, V] = [{B}, {T}, {H}, V], got {tuple(v.shape)}.")
    if K > 128:
        raise ValueError(f"GKA supports a head dim K of at most 128, got K={K}.")
    if num_iter < 1:
        raise ValueError(
            f"`num_iter` must be at least 1, got {num_iter}. Without the ridge solve, GKA reduces to `chunk_simple_gla`.",
        )
    for name, t in (('g', g), ('gk', gk), ('alpha', alpha)):
        if t is not None and t.shape != (B, T, H):
            raise ValueError(f"`{name}` must have shape [B, T, H] = [{B}, {T}, {H}], got {tuple(t.shape)}.")

    N = B
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`. "
                "Please flatten variable-length inputs before processing.",
            )
        N = len(cu_seqlens) - 1
    h_kk0, h_kv0 = initial_state if initial_state is not None else (None, None)
    if h_kk0 is not None and h_kk0.shape != (N, H, K, K):
        raise ValueError(f"`initial_state[0]` must have shape [N, H, K, K] = [{N}, {H}, {K}, {K}], got {tuple(h_kk0.shape)}.")
    if h_kv0 is not None and h_kv0.shape != (N, H, K, V):
        raise ValueError(f"`initial_state[1]` must have shape [N, H, K, V] = [{N}, {H}, {K}, {V}], got {tuple(h_kv0.shape)}.")
    if scale is None:
        scale = K ** -0.5
    return h_kk0, h_kv0, scale


@torch.compiler.disable
def chunk_gka(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    scale: float | None = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
):
    r"""
    Gated KalmaNet: reads out `o_t = scale * U_t^T q_mix_t`, with `q_mix_t = q_t + alpha_t * (x_t - q_t)`,
    and the ridge solution `x_t = (H_t + ridge_ratio * ||H_t||_F * I)^{-1} q_t`.
    The states are `H_t = exp(gk_t) H_{t-1} + k_t k_t^T` and `U_t = exp(g_t) U_{t-1} + k_t v_t^T`,
    stored as `h_kk` and `h_kv`.
    The ridge system is solved with `num_iter` Chebyshev iterations. The sequence is processed in chunks of 64 tokens,
    with the states materialized only at chunk boundaries and the tokens within each chunk solved in parallel.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor, Optional):
            Log-space decay of `U_t`, of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor, Optional):
            Log-space decay of `H_t`, of shape `[B, T, H]`.
            Pass the same tensor as `g` to decay both states together. Default: `None`.
        alpha (torch.Tensor, Optional):
            Mixing weights of shape `[B, T, H]`; the readout query is `q + alpha * (x - q)`.
            If `None`, the readout query is the ridge solution `x`, as with `alpha = 1`. Default: `None`.
        scale (float, Optional):
            Scale applied to the readout. Default: `1 / sqrt(K)`.
        ridge_ratio (float, Optional):
            Sets the ridge strength `lamb_t = ridge_ratio * ||H_t||_F`. Default: `0.02`.
        num_iter (int, Optional):
            Number of Chebyshev iterations, at least 1. Default: `30`.
        initial_state (tuple[torch.Tensor, torch.Tensor], Optional):
            Initial states `(h_kk, h_kv)` of shapes `[N, H, K, K]` and `[N, H, K, V]` in `float32`, for `N` input
            sequences. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final states. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.
        cu_seqlens_cpu (torch.LongTensor, Optional):
            A CPU copy of `cu_seqlens`, used to avoid a device-to-host sync. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (tuple[torch.Tensor, torch.Tensor] | None):
            Final states `(h_kk, h_kv)` of shapes `[N, H, K, K]` and `[N, H, K, V]` in `float32` if
            `output_final_state=True`, else `None`.

    Note:
        For `bfloat16` / `float16` inputs, the backward requires `K >= 64` and `V >= 32`. Below that, the
        `chunk_simple_gla` kernels it relies on hit illegal memory accesses, so a `RuntimeError` is raised instead.
    """
    h_kk0, h_kv0, scale = check_gka_inputs(q, k, v, g, gk, alpha, scale, num_iter, initial_state, cu_seqlens)

    x, h_kk = ChunkGKASolveFunction.apply(
        q,
        k,
        gk,
        ridge_ratio,
        num_iter,
        h_kk0,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
    )
    q_mix = x if alpha is None else (q + alpha[..., None] * (x - q)).to(q.dtype)
    o, h_kv = chunk_simple_gla(
        q=q_mix,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=h_kv0,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    # chunk_simple_gla's backward crashes for 16-bit inputs with K < 64 or V < 32,
    # so SimpleGLABackwardGuard raises a clear error instead
    K, V = q.shape[-1], v.shape[-1]
    if o.requires_grad and o.dtype in (torch.bfloat16, torch.float16) and (K < 64 or V < 32):
        o, h_kv = SimpleGLABackwardGuard.apply(o, h_kv, K, V)
    final_state = (h_kk, h_kv) if output_final_state else None
    return o, final_state
