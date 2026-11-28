# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import IS_NVIDIA_HOPPER, input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_triadic_gdn_fwd_kernel(
    q,
    k,
    v,
    k2,
    q2,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, pid // NV
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_e = tl.arange(0, BE)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * H + i_h) * V + o_v
    p_k2 = k2 + (bos * H + i_h) * E + o_e
    p_q2 = q2 + (bos * H + i_h) * E + o_e
    p_g = g + (bos * H + i_h) * E + o_e
    p_beta = beta + bos * H + i_h
    p_o = o + (bos * H + i_h) * V + o_v

    mask_e = o_e < E
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_e[:, None, None] & mask_k[None, :, None] & mask_v[None, None, :]

    b_h = tl.zeros([BE, BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * E*K*V + o_e[:, None, None] * K*V + o_k[None, :, None] * V + o_v[None, None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_k2 = tl.load(p_k2, mask=mask_e, other=0).to(tl.float32)
        b_q2 = tl.load(p_q2, mask=mask_e, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask_e, other=0).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        b_h *= exp(b_g)[:, None, None]
        # joint key of shape [BE, BK]
        b_kk = b_k2[:, None] * b_k[None, :]
        b_v = b_beta * (b_v - tl.sum(tl.sum(b_h * b_kk[:, :, None], 0), 0))
        b_h += b_kk[:, :, None] * b_v[None, None, :]
        # joint query of shape [BE, BK]
        b_qq = b_q2[:, None] * b_q[None, :]
        b_o = tl.sum(tl.sum(b_h * b_qq[:, :, None], 0), 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_v += H*V
        p_k2 += H*E
        p_q2 += H*E
        p_g += H*E
        p_beta += H
        p_o += H*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * E*K*V + o_e[:, None, None] * K*V + o_k[None, :, None] * V + o_v[None, None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_triadic_gdn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, E = *k.shape, v.shape[-1], k2.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BE = triton.next_power_of_2(E)
    BK = triton.next_power_of_2(K)
    # each program keeps a [BE, BK, BV] state block in registers, with enough warps for at most 128 state values
    # per thread up to 8 warps; Hopper runs best with blocks of up to 32K state values, B200 with up to 16K
    if IS_NVIDIA_HOPPER:
        BV, max_block = (32 if BE <= 2 else 16), 32768
    else:
        BV, max_block = (32 if BE <= 4 else 16), 16384
    BV = min(BV, triton.next_power_of_2(V))
    while BV > 8 and max_block < BE * BK * BV:
        BV //= 2
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    final_state = q.new_empty(N, H, E, K, V, dtype=torch.float32) if output_final_state else None

    grid = (NV * N * H,)
    fused_recurrent_triadic_gdn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        E=E,
        BK=BK,
        BV=BV,
        BE=BE,
        num_warps=min(8, max(4, BE * BK * BV // 4096)),
        num_stages=3,
    )
    return o, final_state


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
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
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        return fused_recurrent_triadic_gdn_fwd(
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
        )

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError("fused_recurrent_triadic_gdn is inference-only; use chunk_triadic_gdn for training.")


def fused_recurrent_triadic_gdn(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Triadic Gated DeltaNet (https://arxiv.org/abs/2609.36529) computed step by step, for inference.

    Each head keeps `E` state slices `S_e` of shape `[K, V]`.
    With the joint key `kappa_t = k2_t ⊗ k_t` and the joint query `rho_t = q2_t ⊗ q_t`, each step computes

        S_e <- exp(g_{t,e}) * S_e
        S   <- S + beta_t * kappa_t (v_t - S^T kappa_t)^T
        o_t  = scale * S^T rho_t

    `E = 1` with `k2 = q2 = 1` is Gated DeltaNet.

    `K` and `V` go up to 256. `E` is tested up to 16, where the tiles are tuned;
    a larger `E` keeps a bigger state block in each program.

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
            Cumulative sequence lengths of shape `[N+1]` used for variable-length inputs,
            consistent with the FlashAttention API. Default: `None`.

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
        raise ValueError(f"fused_recurrent_triadic_gdn supports head dims up to 256, got K={K} and V={v.shape[-1]}.")
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
    return FusedRecurrentFunction.apply(q, k, v, k2, q2, g, beta, scale, initial_state, output_final_state, cu_seqlens)
