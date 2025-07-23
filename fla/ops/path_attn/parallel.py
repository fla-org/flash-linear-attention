# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
from einops import reduce

from fla.ops.attn.parallel import parallel_attn_bwd_preprocess
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.path_attn.cumprod_householder_bwd import chunk_cumprod_householder_bwd_fn
from fla.ops.path_attn.cumprod_householder_fwd import chunk_cumprod_householder_fwd_fn
from fla.ops.path_attn.intra_chunk_preprocess_bwd import intra_chunk_preprocess_bwd_fn
from fla.ops.path_attn.intra_chunk_preprocess_bwd_prepare import intra_chunk_preprocess_bwd_prepare_fn
from fla.ops.path_attn.intra_chunk_preprocess_fwd import intra_chunk_preprocess_fwd_fn
from fla.ops.path_attn.parallel_path_bwd_inter_dkv import parallel_path_bwd_dkv_fn
from fla.ops.path_attn.parallel_path_bwd_inter_dqh import parallel_path_bwd_dq_fn
from fla.ops.path_attn.parallel_path_bwd_intra import parallel_path_bwd_intra_chunk_fn
from fla.ops.path_attn.parallel_path_fwd import parallel_path_fwd_fn
from fla.ops.path_attn.prepare_k_cache import prepare_k_cache_fn
from fla.ops.path_attn.transform_q import transform_q_fwd_fn
from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.solve_tril import solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, input_guard


class ParallelPATHAttentionFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, beta, g, scale, cu_seqlens, use_cache=False):
        g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, output_dtype=torch.float32) if g is not None else None
        BS = 64
        BT = 128 if check_shared_mem('hopper') else 64

        A = chunk_scaled_dot_kkt_fwd(
            k=w,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_size=BS,
            output_dtype=torch.float32
        )
        A = solve_tril(
            A=A,
            cu_seqlens=cu_seqlens,
            output_dtype=w.dtype
        )
        q_new, k_new, w2, o, L, M = intra_chunk_preprocess_fwd_fn(
            q=q,
            k=k,
            v=v,
            w=w,
            beta=beta,
            g_cumsum=g_cumsum,
            A=A,
            scale=scale,
            BT=BS,
            cu_seqlens=cu_seqlens,
        )
        o, L = parallel_path_fwd_fn(
            q=q_new,
            k=k_new,
            v=v,
            L=L,
            w1=w,
            w2=w2,
            M=M,
            o=o,
            g_cumsum=g_cumsum,
            scale=scale,
            cu_seqlens=cu_seqlens,
            BT=BT,
            BS=BS,
        )
        k_cache = prepare_k_cache_fn(k=k_new, w1=w, w2=w2, cu_seqlens=cu_seqlens, BS=BS, use_cache=use_cache)
        ctx.save_for_backward(q, k, v, w, g_cumsum, o, beta, L, A)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o, k_cache

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dk_new):
        if do.shape[-1] > 64 and check_shared_mem('hopper') is False:
            assert False, "Head dimension 128 only supported on Hopper or later. Stay tuned for Ampere support!"
        q, k, v, w, g_cumsum, o, beta, L, A = ctx.saved_tensors
        BT = 128 if check_shared_mem('hopper') else 64
        BS = 64
        S = 512
        cu_seqlens = ctx.cu_seqlens
        delta = parallel_attn_bwd_preprocess(o, do)

        q_new, k_new, h, dA_local, dv, dg_cumsum = intra_chunk_preprocess_bwd_prepare_fn(
            q=q,
            k=k,
            v=v,
            w=w,
            beta=beta,
            g_cumsum=g_cumsum,
            A=A,
            L=L,
            D=delta,
            do=do,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            return_h=False,
        )

        k_new_large, hc_suffix, hc_whole = chunk_cumprod_householder_fwd_fn(
            k=k_new, w1=w, w2=h, S=S, BT=BS, cu_seqlens=cu_seqlens
        )

        q_new_large = transform_q_fwd_fn(q=q_new, w1=w, w2=h, cu_seqlens=cu_seqlens, BT=BT, BS=BS, S=S)

        dk, dv, dg_cumsum3 = parallel_path_bwd_dkv_fn(
            q=q_new_large, k=k_new_large, v=v, g_cumsum=g_cumsum, do=do, dv=dv, dg_cumsum=dg_cumsum,
            hc_whole=hc_whole, scale=ctx.scale, L=L, D=delta,
            cu_seqlens=cu_seqlens,
            S=S, BT=BT, BS=BS
        )
        dq, dhc_whole, dg_cumsum = parallel_path_bwd_dq_fn(
            q=q_new_large, k=k_new_large, v=v, g_cumsum=g_cumsum, do=do, dg_cumsum=dg_cumsum,
            k_new=k_new, w1=w, w2=h,
            hc_whole=hc_whole, scale=ctx.scale, L=L, D=delta,
            cu_seqlens=cu_seqlens,
            S=S, BT=BT, BS=BS
        )
        dw1, dw2, dk = chunk_cumprod_householder_bwd_fn(
            w1=w, w2=h,
            k=k_new, dk=dk, hc_suffix=hc_suffix, dhc_whole=dhc_whole,
            cu_seqlens=cu_seqlens, S=S, BT=BS
        )
        dq, dk, dv, dw1, dw2, dg_cumsum = parallel_path_bwd_intra_chunk_fn(
            q=q_new, k=k_new, v=v, g_cumsum=g_cumsum, w1=w, w2=h,
            L=L, D=delta, scale=ctx.scale, dw1=dw1, dw2=dw2,
            dq=dq, dk=dk, dv=dv, do=do, dg_cumsum=dg_cumsum,
            cu_seqlens=cu_seqlens,
            S=S, BT=BS
        )
        dq, dk, dbeta, dw = intra_chunk_preprocess_bwd_fn(
            q=q, k=k, w=w, beta=beta,
            dq=dq, dk=dk, dw1=dw1, dw2=dw2, dA_local=dA_local,
            A=A, L=L, D=delta, do=do, scale=ctx.scale, cu_seqlens=cu_seqlens
        )
        G = q.shape[-2] // k.shape[-2]
        if G > 1:
            assert dk.dtype == dv.dtype == dw.dtype == dbeta.dtype == torch.float32, 'reduction requires float32'
            dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
            dv = reduce(dv, 'b t (h g) k -> b t h k', g=G, reduction='sum')
            dw = reduce(dw, 'b t (h g) k -> b t h k', g=G, reduction='sum')
            dbeta = reduce(dbeta, 'b t (h g) -> b t h', g=G, reduction='sum')
        if dg_cumsum is not None:
            dg_cumsum = chunk_global_cumsum(dg_cumsum, cu_seqlens=cu_seqlens, reverse=True)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dw.to(w.dtype),
                dbeta.to(beta.dtype),
                dg_cumsum.to(g_cumsum.dtype) if g_cumsum is not None else None,
                None, None, None, None)


@torch.compiler.disable
def parallel_path_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_cache: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, T, H, V]`
        w (torch.Tensor):
            weights of shape `[B, T, H, K]`
        beta (torch.Tensor):
            beta of shape `[B, T, H]`
        g (torch.Tensor):
            g of shape `[B, T, HQ]`
        scale (float):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        use_cache (bool):
            Whether to transform and cache the key values for decoding. Default: `False`.
    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, V]`
        k_cache (torch.Tensor):
            k_cache of shape `[B, T, H, K]`
    """
    if q.shape[-1] > 64 and check_shared_mem('hopper') is False:
        assert False, "Head dimension 128 only supported on Hopper or later. Stay tuned for Ampere support!"
    if scale is None:
        scale = k.shape[-1]**-0.5
    assert q.shape[-1] in [16, 32, 64, 128], "only support head_dim in [16, 32, 64, 128] for now. Stay tuned!"
    assert v.shape[-1] in [16, 32, 64, 128], "only support head_dim in [16, 32, 64, 128] for now. Stay tuned!"
    assert q.shape[-1] == k.shape[-1], 'q, k should have the same head_dim.'
    assert k.shape == w.shape, 'k, w should have the same shape.'
    assert beta.shape[:3] == k.shape[:3], 'beta should have the same number of heads as k'
    if g is not None:
        assert g.shape[:3] == q.shape[:3], 'g should have the same number of heads as q'
    assert q.shape[-2] % k.shape[-2] == 0, 'the number of query heads should be divisible by the number of key heads'
    o, k_cache = ParallelPATHAttentionFunction.apply(q, k, v, w, beta, g, scale, cu_seqlens, use_cache)
    return o, k_cache
