# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_fwd
from fla.utils import contiguous


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    offsets,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if HEAD_FIRST:
        p_q = q + i_nh * T*K + i_k * BK + tl.arange(0, BK)
        p_k = k + i_nh * T*K + i_k * BK + tl.arange(0, BK)
        p_v = v + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + i_nh * T
        p_g = g + i_nh * T
        p_o = o + (i_k * B*H + i_nh) * T*V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
        p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
        p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        if IS_BETA_HEADWISE:
            p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        else:
            p_beta = beta + bos * H + i_h
        p_g = g + bos * H + i_h
        p_o = o + ((i_k * all + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV)

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_g = tl.load(p_g).to(tl.float32)
        b_h *= tl.exp(b_g)
        b_v_minus = tl.sum(b_h * b_k[None, :], axis=1)
        b_v -= b_v_minus
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += K if HEAD_FIRST else H*K
        p_k += K if HEAD_FIRST else H*K
        p_o += V if HEAD_FIRST else H*V
        p_v += V if HEAD_FIRST else H*V
        p_g += 1 if HEAD_FIRST else H
        p_beta += (1 if HEAD_FIRST else H) * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        final_state = None

    grid = (NV, NK, N * H)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        offsets=offsets,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        HEAD_FIRST=head_first,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
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
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True,
        use_qk_l2norm_in_kernel: bool = False
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            head_first=head_first
        )

        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        raise NotImplementedError("Backward pass is not implemented yet and we do not have plans to implement it "
                                  "because we haven't figured out how to compute dg without materializing the full "
                                  "hidden states for all time steps.")


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    use_qk_l2norm_in_kernel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
             g (decays) of shape `[B, H, T]` if `head_first=True` else `(B, T, H)`.
        beta (torch.Tensor):
             betas of shape `[B, H, T]` if `head_first=True` else `(B, T, H)`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, H, device='cuda'))
        >>> beta = torch.rand(B, T, H, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(q, k, v, g, beta,
                                               initial_state=h0,
                                               output_final_state=True,
                                               head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_gated_recurrent_delta_rule(q, k, v, g, beta,
                                                             initial_state=h0,
                                                             output_final_state=True,
                                                             cu_seqlens=cu_seqlens,
                                                             head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel
    )
    return o, final_state
