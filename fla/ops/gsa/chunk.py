# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.gsa.chunk_bwd import chunk_gsa_bwd
from fla.ops.gsa.chunk_fwd import chunk_gsa_fwd
from fla.ops.gsa.gate import fused_additive_gate
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.utils import input_guard


class ChunkGSAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        scale: float,
        hk0: torch.Tensor | None,
        hv0: torch.Tensor | None,
        output_final_state: bool,
        checkpoint_level: int,
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_cpu: torch.LongTensor | None,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if chunk_size is None:
            chunk_size = 64

        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(
                cu_seqlens=cu_seqlens,
                chunk_size=chunk_size,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
        else:
            chunk_indices = None

        # pre-scale gates so both stages use exp2 with the same cumulative values.
        g_org = g
        g = chunk_local_cumsum(
            g=g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        hk, hkt, ok, p, Av, hv, hvt, ov = chunk_gsa_fwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            initial_state=(hk0, hv0),
            output_final_state=output_final_state,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )

        if checkpoint_level >= 1:
            g = g_org
        if checkpoint_level > 1:
            hk, hv = None, None

        ctx.save_for_backward(
            q,
            k,
            v,
            s,
            g,
            ok,
            p,
            Av,
            hk0,
            hv0,
            hk,
            hv,
            chunk_indices,
        )
        ctx.checkpoint_level = checkpoint_level
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_size = chunk_size
        return ov, hkt, hvt

    @staticmethod
    @input_guard
    def backward(ctx, dov, dhkt=None, dhvt=None):
        (
            q,
            k,
            v,
            s,
            g,
            ok,
            p,
            Av,
            hk0,
            hv0,
            hk,
            hv,
            chunk_indices,
        ) = ctx.saved_tensors
        scale = ctx.scale
        cu_seqlens = ctx.cu_seqlens
        chunk_size = ctx.chunk_size

        if ctx.checkpoint_level >= 1:
            g = chunk_local_cumsum(
                g=g,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        dq, dk, dv, ds, dg, dhk0, dhv0 = chunk_gsa_bwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            ok=ok,
            p=p,
            A=Av,
            h=(hk, hv),
            initial_state=(hk0, hv0),
            scale=scale,
            do=dov,
            dht=(dhkt, dhvt),
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        return dq, dk, dv, ds, dg, None, dhk0, dhv0, None, None, None, None, None


@torch.compiler.disable
def chunk_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    checkpoint_level: int = 2,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA is performed if `H` is not equal to `HQ`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        s (torch.Tensor):
            slot representations of shape `[B, T, H, M]`.
        g (Optional[torch.Tensor]):
            Forget gates of shape `[B, T, H, M]` applied along the slot dimension.
            If not provided, this function is equivalent to vanilla ABC. Default: `None`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state tuple, having tensors of shape `[N, H, K, M]` and `[N, H, M, V]`.
            Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memory and do more recomputations during backward.
            Default: `2`.
            - Level `0`: no memory saved, no recomputation.
            - Level `1`: recompute the fp32 cumulative values during backward.
            - Level `2`: recompute cumulative gates and chunk states during backward.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cu_seqlens_cpu (torch.LongTensor):
            CPU copy of `cu_seqlens` to avoid unnecessary device synchronization. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
        final_state (Tuple[torch.Tensor]):
            Final state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` if `output_final_state=True`.
            Both entries are `None` otherwise.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gsa import chunk_gsa
        # inputs with equal lengths
        >>> B, T, H, K, V, M = 4, 2048, 4, 512, 512, 64
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> s = torch.randn(B, T, H, M, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, M, device='cuda'))
        >>> h0 = (torch.randn(B, H, K, M, device='cuda'), torch.randn(B, H, M, V, device='cuda'))
        >>> o, (hk, hv) = chunk_gsa(
            q, k, v, s, g,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, s, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, s, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, (hk_var, hv_var) = chunk_gsa(
            q, k, v, s, g,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert hk.allclose(hk_var)
        >>> assert hv.allclose(hv_var)
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    chunk_size = kwargs.pop('chunk_size', None)
    if chunk_size is not None and (chunk_size <= 0 or chunk_size & (chunk_size - 1)):
        raise ValueError(f"`chunk_size` must be a power of 2, got {chunk_size}.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state[0].shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state[0].shape[0]}.",
            )
    assert checkpoint_level in [0, 1, 2]
    if g is None:
        s, g, _ = fused_additive_gate(
            x=s,
            cu_seqlens=cu_seqlens,
            output_dtype=k.dtype,
            gate_dtype=torch.float32,
        )
    if scale is None:
        scale = q.shape[-1] ** -0.5

    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    H, HQ = k.shape[2], q.shape[2]
    if HQ < H or HQ % H:
        raise ValueError(f"The number of query heads ({HQ}) must be a multiple of the number of key heads ({H}).")
    NG = HQ // H
    if NG > 1:
        k, v, s, g = (x.repeat_interleave(NG, dim=2) for x in (k, v, s, g))
        hk0, hv0 = (x.repeat_interleave(NG, dim=1) if x is not None else None for x in (hk0, hv0))
    o, *final_state = ChunkGSAFunction.apply(
        q,
        k,
        v,
        s,
        g,
        scale,
        hk0,
        hv0,
        output_final_state,
        checkpoint_level,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_size,
    )
    if NG > 1 and output_final_state:
        final_state = [x[:, ::NG].contiguous() for x in final_state]
    return o, final_state
