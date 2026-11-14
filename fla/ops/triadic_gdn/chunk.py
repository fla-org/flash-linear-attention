# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.triadic_gdn.chunk_bwd import chunk_triadic_gdn_bwd
from fla.ops.triadic_gdn.chunk_fwd import chunk_triadic_gdn_fwd
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class ChunkTriadicGDNFunction(torch.autograd.Function):

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
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        disable_recompute: bool = False,
        chunk_size: int = 64,
    ):
        if cu_seqlens is not None and chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        g, Ai, Aqk, h, v_new, o, final_state = chunk_triadic_gdn_fwd(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            save_states=disable_recompute,
        )
        ctx.save_for_backward(q, k, v, k2, q2, g, beta, Ai, Aqk, h, v_new, initial_state, cu_seqlens, chunk_indices)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        q, k, v, k2, q2, g, beta, Ai, Aqk, h, v_new, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors
        dq, dk, dv, dk2, dq2, dg, dbeta, dh0 = chunk_triadic_gdn_bwd(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            g=g,
            beta=beta,
            Ai=Ai,
            Aqk=Aqk,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            h=h,
            u=v_new,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
        )
        return (
            dq.to(q), dk.to(k), dv.to(v), dk2.to(k2), dq2.to(q2), dg.to(g), dbeta.to(beta),
            None, dh0, None, None, None, None, None,
        )


@torch.compiler.disable
def chunk_triadic_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    disable_recompute: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Triadic Gated DeltaNet (https://arxiv.org/abs/2609.36529) in chunkwise-parallel form, for training and prefill.

    Each head keeps `E` state slices `S_e` of shape `[K, V]`.
    With the joint key `kappa_t = k2_t ⊗ k_t` and the joint query `rho_t = q2_t ⊗ q_t`, each step computes

        S_e <- exp(g_{t,e}) * S_e
        S   <- S + beta_t * kappa_t (v_t - S^T kappa_t)^T
        o_t  = scale * S^T rho_t

    `E = 1` with `k2 = q2 = 1` is Gated DeltaNet.

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
        g (torch.Tensor):
            Per-slice decays in log space of shape `[B, T, H, E]`.
        beta (torch.Tensor):
            Write strengths of shape `[B, T, H]`.
        scale (float, Optional):
            Scale factor of the output. Default: `1 / sqrt(K)`.
        initial_state (torch.Tensor, Optional):
            Initial state of shape `[N, H, E, K, V]` for `N` input sequences. Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final state of shape `[N, H, E, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.
        chunk_indices (torch.LongTensor, Optional):
            Chunk indices precomputed from `cu_seqlens`. Default: `None`.
        disable_recompute (bool, Optional):
            Whether to save the chunk states of the forward pass instead of recomputing them in the backward pass,
            which is faster at the cost of `E * K * V` values per chunk and head. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, E, K, V]` in float32 if `output_final_state=True` else `None`.
    """
    B, T, H, K, E = *k.shape, k2.shape[-1]
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {list(q.shape)} and {list(k.shape)}.")
    if v.shape[:3] != (B, T, H):
        raise ValueError(f"v must have shape [B, T, H, V]=[{B}, {T}, {H}, V], got {list(v.shape)}.")
    if k2.shape != (B, T, H, E) or q2.shape != (B, T, H, E) or g.shape != (B, T, H, E):
        raise ValueError(
            f"k2, q2 and g must have shape [B, T, H, E]={[B, T, H, E]}, "
            f"got {list(k2.shape)}, {list(q2.shape)} and {list(g.shape)}.",
        )
    if beta.shape != (B, T, H):
        raise ValueError(f"beta must have shape [B, T, H]={[B, T, H]}, got {list(beta.shape)}.")
    if K > 256 or v.shape[-1] > 256:
        raise ValueError(f"chunk_triadic_gdn supports head dims up to 256, got K={K} and V={v.shape[-1]}.")
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = K ** -0.5
    return ChunkTriadicGDNFunction.apply(
        q, k, v, k2, q2, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_indices, disable_recompute,
    )
