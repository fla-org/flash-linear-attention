# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_h import chunk_bwd_dh
from fla.ops.mesa_net.chunk_cg_solver_bwd import chunk_mesa_cg_bwd
from fla.ops.mesa_net.chunk_cg_solver_fwd import chunk_mesa_cg_fwd
from fla.ops.mesa_net.chunk_h_fwd import chunk_mesa_fwd_h
from fla.ops.mesa_net.chunk_h_kk_intra_bwd import chunk_mesa_net_h_kk_bwd_intra_fn
from fla.ops.mesa_net.chunk_h_kv_intra_bwd import chunk_mesa_net_h_kv_bwd_intra_fn
from fla.ops.utils import chunk_local_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_fwd_mesa_net_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lamb: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_CG_iteration: int = 30,
    chunk_size: int = 64,
    h_kk_init: Optional[torch.Tensor] = None,
    h_kv_init: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> torch.Tensor:

    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens) if g is not None else None
    h_kk, h_kv, h_kk_final, h_kv_final = chunk_mesa_fwd_h(
        k=k,
        v=v,
        g=g,
        beta=beta,
        h_init=h_kk_init,
        h_kv_init=h_kv_init,
        output_final_state=output_final_state,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    q_star, o = chunk_mesa_cg_fwd(
        q=q,
        k=k,
        h=h_kk,
        h_kv=h_kv,
        v=v,
        g_local_cumsum=g,
        beta=beta,
        lamb=lamb,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        max_CG_iteration=max_CG_iteration
    )
    return g, q_star, o, (h_kk_final, h_kv_final)


def chunk_fwd_mesa_net_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lamb: torch.Tensor,
    q_star: torch.Tensor,  # should be cached in the forward pass
    do: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_CG_iteration: int = 30,
    chunk_size: int = 64,
    h_kk_init: Optional[torch.Tensor] = None,
    h_kv_init: Optional[torch.Tensor] = None,
    dh_kv_final: Optional[torch.Tensor] = None,
    dh_kk_final: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # recompute the hidden states, which is quite cheap
    h_kk, h_kv, _, _ = chunk_mesa_fwd_h(
        k=k,
        v=v,
        g=g,
        beta=beta,
        h_init=h_kk_init,
        h_kv_init=h_kv_init,
        output_final_state=False,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    dh_kv, dh0_kv = chunk_bwd_dh(
        q=q_star,
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        do=do,
        h0=h_kv_init,
        dht=dh_kv_final,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        scale=1,
    )
    dq, dk_beta, dv, dg = chunk_mesa_net_h_kv_bwd_intra_fn(
        q_star=q_star,
        k=k,
        v=v,
        beta=beta,
        h_kv=h_kv,
        dh_kv=dh_kv,
        g=g,
        do=do,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    dq = chunk_mesa_cg_bwd(
        dq=dq,
        k=k,
        h=h_kk,
        g_local_cumsum=g,
        beta=beta,
        lamb=lamb,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        max_CG_iteration=max_CG_iteration,
        output_dtype=torch.float16
    )
    dh_kk, dh0_kk = chunk_bwd_dh(
        q=dq,
        k=k,
        v=k,
        g=g,
        gk=None,
        gv=None,
        do=q_star,
        h0=h_kk_init,
        dht=-dh_kk_final if dh_kk_final is not None else None,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        scale=1,
    )
    dk, dg2, dlamb, dbeta = chunk_mesa_net_h_kk_bwd_intra_fn(
        k=k,
        g=g,
        beta=beta,
        h=h_kk,
        dh=dh_kk,
        dk_beta=dk_beta,
        q_star=q_star,
        dq=dq,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    dg.add_(dg2)
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens).to(g)
    return dq, dk, dv, dg, dbeta, dlamb, -dh0_kk if dh0_kk is not None else None, dh0_kv if dh0_kv is not None else None


class ChunkMesaNetFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        lamb,
        cu_seqlens,
        max_CG_iteration,
        h_kk_init,
        h_kv_init,
        output_final_state,
        use_qk_l2norm_in_kernel
    ):
        chunk_size = 64

        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q, output_dtype=torch.float16)
            k, k_rstd = l2norm_fwd(k, output_dtype=torch.float16)
        else:
            q_rstd, k_rstd = None, None
            q = q.to(torch.float16)
            k = k.to(torch.float16)

        g_cumsum, q_star, o, (h_kk_final, h_kv_final) = chunk_fwd_mesa_net_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            lamb=lamb,
            cu_seqlens=cu_seqlens,
            max_CG_iteration=max_CG_iteration,
            chunk_size=chunk_size,
            h_kk_init=h_kk_init,
            h_kv_init=h_kv_init,
            output_final_state=output_final_state,
        )
        ctx.max_CG_iteration = max_CG_iteration
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_cumsum, beta, lamb, h_kk_init, h_kv_init, q_star, o)
        return o, h_kk_final, h_kv_final

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dh_kk_final=None, dh_kv_final=None):
        q, q_rstd, k, k_rstd, v, g, beta, lamb, h_kk_init, h_kv_init, q_star, o = ctx.saved_tensors

        max_CG_iteration = ctx.max_CG_iteration
        chunk_size = ctx.chunk_size
        cu_seqlens = ctx.cu_seqlens
        dq, dk, dv, dg, dbeta, dlamb, dh0_kk, dh0_kv = chunk_fwd_mesa_net_bwd(
            q=q, k=k, v=v, g=g, beta=beta, lamb=lamb, q_star=q_star, do=do,
            cu_seqlens=cu_seqlens, max_CG_iteration=max_CG_iteration, chunk_size=chunk_size,
            h_kk_init=h_kk_init, h_kv_init=h_kv_init, dh_kv_final=dh_kv_final, dh_kk_final=dh_kk_final
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq, dk, dv.to(v), dg.to(g), dbeta.to(beta), dlamb.to(lamb), None, None, dh0_kk, dh0_kv, None, None


@torch.compiler.disable
def chunk_mesa_net(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lamb: torch.Tensor,
    h_kk_init: Optional[torch.Tensor] = None,
    h_kv_init: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    max_CG_iteration: int = 30,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`. Should be l2-normalized before passing in.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            decay factors of shape `[B, T, H]`. Note that `g` should be in log space, that is, `g = log(decay_factor) < 0`.
            Recommended input dtype: `torch.float32`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`. Recommended input dtype: `torch.float32`.
        lamb (torch.Tensor):
            lambdas of shape `[B, T, H]`. Recommended input dtype: `torch.float32`.
        h_kk_init (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        h_kv_init (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        max_CG_iteration (int):
            Maximum number of conjugate gradient iterations for solving the linear system. Default: `30`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Do l2 normalization on Q and K in the kernel for saving GPU memory. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        (final_states_kk, final_states_kv) (Tuple[torch.Tensor, torch.Tensor]):
            Final states of shape `[N, H, K, K]` and `[N, H, K, V]` if `output_final_state=True` else `(None, None)`.
            Recall that MesaNet has two states, `h_kk` and `h_kv`!

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.mesa_net import chunk_mesa_net
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 16, 128, 128
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device='cuda'))
        >>> beta = torch.rand(B, T, H, dtype=torch.float32, device='cuda').sigmoid()
        # lower bound is 0.25 for numerical stability
        >>> lamb = F.softplus(torch.rand(H, K, dtype=torch.float32, device='cuda')) + 0.25
        >>> init_state_kk = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')
        >>> init_state_kv = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')
        >>> o, (final_state_kk, final_state_kv) = chunk_mesa_net(
            q, k, v, beta, lamb,
            h_kk_init=init_state_kk,
            h_kv_init=init_state_kv,
            max_CG_iteration=30,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, (final_state_kk_var, final_state_kv_var) = chunk_mesa_net(
            q, k, v, beta, lamb,
            h_kk_init=init_state_kk,
            h_kv_init=init_state_kv,
            max_CG_iteration=30,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    B, T, H, K = q.shape
    assert k.shape == (B, T, H, K), "k must be of shape (batch size, seq len, num head, head dim)."
    assert v.shape == (B, T, H, K), "v must be of shape (batch size, seq len, num head, head dim)."
    assert g.shape == (B, T, H), "g must be of shape (batch size, seq len, num head)."
    assert beta.shape == (B, T, H), "beta must be of shape (batch size, seq len, num head)."
    assert lamb.shape == (H, K), "lamb must be of shape (num head, key dim)."

    if h_kv_init is not None:
        assert h_kv_init.dtype == torch.float32, "h_kv_init must be in float32."
        if cu_seqlens is None:
            assert h_kv_init.shape == (B, H, K, K), "h_kv_init must be of shape (batch size, num head, head dim, head dim)."
    if h_kk_init is not None:
        assert h_kk_init.dtype == torch.float32, "h_kk_init must be in float32."
        if cu_seqlens is None:
            assert h_kk_init.shape == (B, H, K, K), "h_kk_init must be of shape (batch size, num head, head dim, head dim)."

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if h_kk_init is not None and h_kk_init.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {h_kk_init.shape[0]}."
            )
        if h_kv_init is not None and h_kv_init.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {h_kv_init.shape[0]}."
            )
    o, final_state_kk, final_state_kv = ChunkMesaNetFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        lamb,
        cu_seqlens,
        max_CG_iteration,
        h_kk_init,
        h_kv_init,
        output_final_state,
        use_qk_l2norm_in_kernel
    )
    return o, final_state_kk, final_state_kv
