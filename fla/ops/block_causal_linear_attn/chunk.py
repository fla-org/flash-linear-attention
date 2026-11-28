# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Block-causal (gated) linear attention: every token reads the state at the end of its own block,
#     S_t = exp(g_t) * S_{t-1} + k_t v_t^T,    o_t = scale * q_t^T S_{e(t)},
# where e(t) is the last position of the block containing t. Without gates this is linear attention under
# a block-causal mask: tokens see their whole block and all previous blocks.
#
# Only the states at block boundaries are needed, so they come from the chunked state kernels with
# `split_size=block_size`, and the rest is batched matmuls over blocks. In the backward pass, the gradient
# `scale * Q_b^T dO_b` of the state at the end of block b is injected token by token at its own position
# with q_t rescaled by exp(G_e - G_t). This is wrong inside the block but exact at the block starts.

import torch
import torch.nn.functional as F

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.utils import chunk_global_cumsum, chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_token_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class BlockLayout:
    """Maps tokens `[B, T, ...]` to zero-padded blocks `[NB, BS, ...]` and back, sequence by sequence."""

    def __init__(self, B: int, T: int, block_size: int, cu_seqlens: torch.LongTensor | None):
        self.T, self.BS = T, block_size
        if cu_seqlens is None:
            NS = -(-T // block_size)
            self.NB, self.index = B * NS, None
            self.offsets = torch.arange(0, B + 1, device='cpu') * NS
        else:
            self.offsets = prepare_chunk_offsets(cu_seqlens, block_size)
            seq, pos = prepare_token_indices(cu_seqlens).unbind(-1)
            self.NB = int(self.offsets[-1])
            self.index = self.offsets[seq] * block_size + pos
        # the last block of every sequence, whose end state is the final state
        self.last = self.offsets[1:].to(cu_seqlens.device if cu_seqlens is not None else 'cpu') - 1

    def to_blocks(self, x: torch.Tensor) -> torch.Tensor:
        if self.index is None:
            x = F.pad(x, (0, 0) * (x.ndim - 2) + (0, self.NB // x.shape[0] * self.BS - self.T))
            return x.reshape(self.NB, self.BS, *x.shape[2:])
        y = x.new_zeros(self.NB * self.BS, *x.shape[2:]).index_copy_(0, self.index, x[0])
        return y.view(self.NB, self.BS, *x.shape[2:])

    def from_blocks(self, x: torch.Tensor, B: int) -> torch.Tensor:
        if self.index is None:
            return x.reshape(B, -1, *x.shape[2:])[:, :self.T]
        return x.flatten(0, 1)[self.index].unsqueeze(0)

    def end_states(self, h: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        """States at block ends `[NB, ...]` from states at block starts `h` `[NB, ...]` and the final states `ht`."""
        s = torch.empty_like(h)
        s[:-1] = h[1:]
        s[self.last.to(h.device)] = ht.to(s.dtype)
        return s


def exclusive_reverse_cumsum(g: torch.Tensor) -> torch.Tensor:
    """sum_{t < s <= e} g_s within each block `[NB, BS, H]`, i.e., the log decay from t to the block end."""
    return g.flip(1).cumsum(1).flip(1) - g


def chunk_block_causal_linear_attn_fwd(q, k, v, g, scale, initial_state, layout, cu_seqlens, chunk_size, chunk_indices):
    g2 = None
    if g is not None:
        g2 = chunk_local_cumsum(g, chunk_size, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g2,
        h0=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        split_size=layout.BS,
        states_in_fp32=True,
    )
    s = layout.end_states(h.flatten(0, 1), ht)
    o = torch.einsum('nshk,nhkv->nshv', layout.to_blocks(q), s.to(q.dtype)) * scale
    return layout.from_blocks(o, q.shape[0]), s, ht


def chunk_block_causal_linear_attn_bwd(q, k, v, g, s, ht, scale, initial_state, do, dht, layout, cu_seqlens, chunk_size,
                                       chunk_indices):
    B, H = q.shape[0], q.shape[2]
    qb, kb, vb, dob = map(layout.to_blocks, (q, k, v, do))
    # exp(G_e - G_t): the decay from each token to the end of its block
    decay = exclusive_reverse_cumsum(layout.to_blocks(g.float())).exp() if g is not None else None

    dq = torch.einsum('nshv,nhkv->nshk', dob, s.to(do.dtype)) * scale

    g2, qd = None, q
    if g is not None:
        g2 = chunk_local_cumsum(g, chunk_size, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
        qd = layout.from_blocks((qb * decay[..., None]).to(q.dtype), B)
    dh, dh0 = chunk_bwd_dh(
        q=qd,
        k=k,
        v=v,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        g=g2,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        split_size=layout.BS,
        states_in_fp32=True,
    )
    # `dh` holds the gradients after the first chunk of every block, so step back over that chunk to the block start,
    # where the rescaled injections of the whole block are exact; this is the gradient of the end state of the
    # previous block from all later blocks
    d0 = torch.einsum('nshk,nshv->nhkv', qb[:, :chunk_size].float(), dob[:, :chunk_size].float()) * scale
    if g is not None:
        gb = layout.to_blocks(g.float())
        dh = gb[:, :chunk_size].sum(1).exp()[..., None, None] * dh.flatten(0, 1) + gb.sum(1).exp()[..., None, None] * d0
    else:
        dh = dh.flatten(0, 1) + d0
    ds = layout.end_states(dh, dht if dht is not None else torch.zeros_like(dh[:len(layout.last)]))
    # plus the reads of the block itself
    ds = ds + torch.einsum('nshk,nshv->nhkv', qb.float(), dob.float()) * scale

    dkb = torch.einsum('nshv,nhkv->nshk', vb, ds.to(v.dtype))
    dvb = torch.einsum('nshk,nhkv->nshv', kb, ds.to(k.dtype))
    if g is not None:
        dkb, dvb = dkb * decay[..., None], dvb * decay[..., None]
    dk, dv = layout.from_blocks(dkb, B), layout.from_blocks(dvb, B)

    dg = None
    if g is not None:
        # dg_t = sum_{t' >= t} (<dS_t', S_t' - k_t' v_t'^T>), which leaves -k . dk per token, <dS_e, S_e> = sum q . dq
        # over the block at every block end, and <dht, ht> at the end of every sequence
        dg = chunk_global_cumsum(-(k.float() * dk.float()).sum(-1), reverse=True, cu_seqlens=cu_seqlens)
        # [1, NB, H] reverse cumsum over the blocks of every sequence
        dg_blocks = chunk_global_cumsum(
            (qb.float() * dq.float()).sum((1, -1)).view(1 if cu_seqlens is not None else B, -1, H),
            reverse=True,
            cu_seqlens=layout.offsets.to(q.device) if cu_seqlens is not None else None,
        )
        dg = dg + layout.from_blocks(dg_blocks.view(layout.NB, 1, H).expand(-1, layout.BS, -1), B)
        if dht is not None:
            dg_final = (dht * ht).sum((-2, -1))
            if cu_seqlens is not None:
                dg = dg + dg_final[prepare_token_indices(cu_seqlens)[:, 0]].unsqueeze(0)
            else:
                dg = dg + dg_final[:, None]
    dq = layout.from_blocks(dq, B)
    return dq, dk, dv, dg, dh0


class ChunkBlockCausalLinearAttnFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, block_size, scale, initial_state, output_final_state, cu_seqlens):
        chunk_size = 64 if block_size % 64 == 0 else 32 if block_size % 32 == 0 else 16
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
        layout = BlockLayout(q.shape[0], q.shape[1], block_size, cu_seqlens)
        o, s, ht = chunk_block_causal_linear_attn_fwd(
            q, k, v, g, scale, initial_state, layout, cu_seqlens, chunk_size, chunk_indices,
        )
        ctx.save_for_backward(q, k, v, g, s, ht, initial_state)
        ctx.layout, ctx.scale, ctx.cu_seqlens = layout, scale, cu_seqlens
        ctx.chunk_size, ctx.chunk_indices = chunk_size, chunk_indices
        return o.to(q.dtype), (ht if output_final_state else None)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, g, s, ht, initial_state = ctx.saved_tensors
        dq, dk, dv, dg, dh0 = chunk_block_causal_linear_attn_bwd(
            q, k, v, g, s, ht, ctx.scale, initial_state, do, dht, ctx.layout, ctx.cu_seqlens, ctx.chunk_size,
            ctx.chunk_indices,
        )
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g) if g is not None else None, None, None, dh0, None, None


@torch.compiler.disable
def chunk_block_causal_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    block_size: int = 64,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Linear attention under a block-causal mask: every token attends to all tokens of its own block and of the
    previous blocks. With forget gates, every token reads the recurrent state at the end of its block:

    .. math::
        S_t = \exp(g_t) S_{t-1} + k_t v_t^\top, \qquad o_t = \text{scale} \cdot q_t^\top S_{e(t)},

    where :math:`e(t)` is the last position of the block containing :math:`t`. Blocks are counted from the start
    of every sequence, so the last block of a sequence may be shorter than `block_size`.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (Optional[torch.Tensor]):
            Forget gates in log space of shape `[B, T, H]`. Default: `None`.
        block_size (int):
            Number of tokens per block, a multiple of 16. Default: `64`.
        scale (Optional[float]):
            Scale factor for the attention scores. If not provided, it will default to `1 / sqrt(K)`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    if block_size <= 0 or block_size % 16 != 0:
        raise ValueError(f"`block_size` must be a positive multiple of 16, got {block_size}.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return ChunkBlockCausalLinearAttnFunction.apply(
        q, k, v, g, block_size, scale, initial_state, output_final_state, cu_seqlens,
    )
