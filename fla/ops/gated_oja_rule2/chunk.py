# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.modules.norm.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.gated_oja_rule2.chunk_h import chunk_oja2_bwd_dhu, chunk_oja2_bwd_dvwg_h, chunk_oja2_fwd_h
from fla.ops.gated_oja_rule2.chunk_kkt import chunk_oja2_kkt_bwd_gk, chunk_oja2_kkt_fwd
from fla.ops.gated_oja_rule2.chunk_o import (
    chunk_oja2_bwd_dA,
    chunk_oja2_bwd_dqk,
    chunk_oja2_bwd_dv_o,
    chunk_oja2_fwd_o,
)
from fla.ops.gated_oja_rule2.wy_fast import prepare_wy_repr_bwd_oja2, recompute_w_u_fwd_oja2
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_oja2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    gv = chunk_local_cumsum(gv, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A = chunk_oja2_kkt_fwd(
        k=v,
        gk=gv,
        b=b,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u, vg = recompute_w_u_fwd_oja2(
        k=k,
        v=v,
        b_gate=b,
        c_gate=c,
        A=A,
        gv=gv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, k_new, final_state = chunk_oja2_fwd_h(
        v=vg,
        w=w,
        u=u,
        gv=gv,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    _, o = chunk_oja2_fwd_o(
        q=q,
        k=k_new,
        v=v,
        h=h,
        gv=gv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return gv, o, A, final_state


def chunk_oja2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    A: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    w, u, vg = recompute_w_u_fwd_oja2(
        k=k,
        v=v,
        b_gate=b,
        c_gate=c,
        A=A,
        gv=gv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    h, k_new, _ = chunk_oja2_fwd_h(
        v=vg,
        w=w,
        u=u,
        gv=gv,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    dAqk = chunk_oja2_bwd_dA(
        v=v,
        gv=gv,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    Aqk, dq, dk_new = chunk_oja2_bwd_dqk(
        q=q,
        k=k_new,
        h=h,
        gv=gv,
        dA=dAqk,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    dh, dh0, dk_new = chunk_oja2_bwd_dhu(
        q=q,
        vg=vg,
        w=w,
        gv=gv,
        h0=initial_state,
        dht=dht,
        do=do,
        dk=dk_new,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        states_in_fp32=False,
        chunk_size=chunk_size,
    )

    dv, dw, dgv_last = chunk_oja2_bwd_dvwg_h(
        k=k_new,
        v=v,
        gv=gv,
        h=h,
        dh=dh,
        dk=dk_new,
        dgk=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    dv, dgv1 = chunk_oja2_bwd_dv_o(
        v=v,
        gv=gv,
        o=o,
        A=Aqk,
        dv=dv,
        do=do,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    dk, dv1, db, dc, dgv2, dAvv = prepare_wy_repr_bwd_oja2(
        k=k,
        v=v,
        b_gate=b,
        c_gate=c,
        A=A,
        dw=dw,
        du=dk_new,
        gv=gv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    dv2, dgv3, db2 = chunk_oja2_kkt_bwd_gk(
        k=v,
        g=gv,
        b=b,
        dA=dAvv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    dv = dv.add_(dv1).add_(dv2)
    db = db.add_(db2)
    dgv = dgv_last.add_(chunk_local_cumsum(
        g=dgv1.add_(dgv2).add_(dgv3),
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    ))
    return dq, dk, dv, db, dc, dgv, dh0


class ChunkOja2Function(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gv: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        use_q_l2norm: bool = False,
        use_k_l2norm: bool = False,
        chunk_size: int = 64,
    ):
        q_rstd, k_rstd = None, None
        if use_q_l2norm:
            q, q_rstd = l2norm_fwd(q)
        if use_k_l2norm:
            k, k_rstd = l2norm_fwd(k)

        chunk_indices = None
        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        gv, o, A, final_state = chunk_oja2_fwd(
            q=q,
            k=k,
            v=v,
            gv=gv,
            b=b,
            c=c,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, gv, b, c, A, o, initial_state, cu_seqlens, chunk_indices)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.use_q_l2norm = use_q_l2norm
        ctx.use_k_l2norm = use_k_l2norm
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, q_rstd, k, k_rstd, v, gv, b, c, A, o, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors
        dq, dk, dv, db, dc, dgv, dh0 = chunk_oja2_bwd(
            q=q,
            k=k,
            v=v,
            gv=gv,
            b=b,
            c=c,
            A=A,
            o=o,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
        )

        if ctx.use_q_l2norm:
            dq = l2norm_bwd(q, q_rstd, dq)
        if ctx.use_k_l2norm:
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dgv.to(gv), db.to(b), dc.to(c), None, dh0, None, None, None, None, None, None


@torch.compiler.disable
def chunk_gated_oja_rule2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_q_l2norm: bool = False,
    use_k_l2norm: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Chunkwise Oja2: the Oja rule with decoupled erase and write gates.

    The recurrence acts on a slot memory `S` of shape `[K, M]`:

        S_t = S_{t-1} Diag(exp(gv_t)) (I - (b_t * v_t) v_t^T) + (c_t * k_t) v_t^T
        o_t = S_t^T q_t

    where `*` is the elementwise product, `v_t` is the slot code, `b_t` is the per-slot erase gate on the `M` axis, and
    `c_t` is the per-channel write gate on the `K` axis. Setting `b_t = c_t = beta` (scalar) recovers `gated_oja_rule`.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            slot code of shape `[B, T, H, M]`.
        gv (torch.Tensor):
            log-decay of shape `[B, T, H, M]`.
        b (torch.Tensor):
            per-slot erase gate of shape `[B, T, H, M]`. Typical range: `[0, 1]`.
        c (torch.Tensor):
            per-channel write gate of shape `[B, T, H, K]`. Typical range: `[0, 1]`.
        scale (float, Optional):
            Scale factor for the attention scores. If not provided, it defaults to `1 / sqrt(K)`.
        initial_state (torch.Tensor, Optional):
            Initial state of shape `[N, H, K, M]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final state of shape `[N, H, K, M]`. Default: `False`.
        use_q_l2norm (bool, Optional):
            Whether to L2-normalize `q` inside the kernel. Default: `False`.
        use_k_l2norm (bool, Optional):
            Whether to L2-normalize `k` inside the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training.
            Requires `B = 1`. Default: `None`.
        cu_seqlens_cpu (torch.LongTensor, Optional):
            CPU copy of `cu_seqlens`, used to avoid a device sync. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, M]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, M]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from fla.ops.gated_oja_rule2 import chunk_gated_oja_rule2
        >>> B, T, H, K, M = 2, 1024, 4, 128, 128
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = F.normalize(torch.randn(B, T, H, M, dtype=torch.float32, device='cuda'), dim=-1)
        >>> gv = F.logsigmoid(torch.randn(B, T, H, M, dtype=torch.float32, device='cuda'))
        >>> b = torch.rand(B, T, H, M, dtype=torch.bfloat16, device='cuda')
        >>> c = torch.rand(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_oja_rule2(q, k, v, gv, b, c, use_q_l2norm=True, use_k_l2norm=True, output_final_state=True)
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if 'use_qk_l2norm_in_kernel' in kwargs and (not use_q_l2norm and not use_k_l2norm):
        use_q_l2norm = True
        use_k_l2norm = True

    chunk_size = kwargs.pop('chunk_size', 64)
    if chunk_size not in (16, 32, 64):
        raise ValueError(f"`chunk_size` must be 16, 32, or 64, got {chunk_size}.")

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    assert q.shape == k.shape, "q and k must have the same shape."
    assert v.shape == gv.shape == b.shape, "v, gv and b must have the same shape."
    assert c.shape == k.shape, "c and k must have the same shape."
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkOja2Function.apply(
        q,
        k,
        v,
        gv,
        b,
        c,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_q_l2norm,
        use_k_l2norm,
        chunk_size,
    )
    return o, final_state
