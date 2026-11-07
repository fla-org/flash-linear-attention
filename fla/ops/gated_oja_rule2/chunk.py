# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.gated_oja_rule2.chunk_bwd import chunk_oja2_bwd
from fla.ops.gated_oja_rule2.chunk_fwd import chunk_oja2_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


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
        initial_state: torch.Tensor | None,
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
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Chunkwise Oja2: the Oja rule with decoupled erase and write gates.

    See ``fla/ops/gated_oja_rule2/README.md`` for the recurrence and supported numerical contract.

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

    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if kwargs.get('use_qk_l2norm_in_kernel', False) and not (use_q_l2norm or use_k_l2norm):
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
    assert q.shape[:3] == v.shape[:3], "q, k and v must have matching batch, sequence and head dimensions."
    assert q.shape == k.shape, "q and k must have the same shape."
    assert v.shape == gv.shape == b.shape, "v, gv and b must have the same shape."
    assert c.shape == k.shape, "c and k must have the same shape."
    if q.shape[-1] <= 0 or not 0 < v.shape[-1] <= 256:
        raise ValueError("The key dimension must be positive and the slot dimension must be between 1 and 256.")
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
