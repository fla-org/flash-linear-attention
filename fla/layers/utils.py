# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Code is adapted from flash-attn.bert_padding.py

from typing import Tuple

import torch
from einops import rearrange, repeat

from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
from fla.utils import tensor_cache


class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, indices):
        ctx.save_for_backward(indices)
        assert x.ndim >= 2
        ctx.first_axis_dim, other_shape = x.shape[0], x.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return x[indices]
        return torch.gather(
            rearrange(x, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, do):
        (indices,) = ctx.saved_tensors
        assert do.ndim >= 2
        other_shape = do.shape[1:]
        do = rearrange(do, "b ... -> b (...)")
        dx = torch.zeros(
            [ctx.first_axis_dim, do.shape[1]],
            device=do.device,
            dtype=do.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # dx[indices] = do
        dx.scatter_(0, repeat(indices, "z -> z d", d=do.shape[1]), do)
        return dx.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert x.ndim >= 2
        y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
        # TODO [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        y[indices] = x
        # y.scatter_(0, repeat(indices, 'z -> z d', d=x.shape[1]), x)
        return y

    @staticmethod
    def backward(ctx, do):
        (indices,) = ctx.saved_tensors
        # TODO [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        dx = do[indices]
        # dx = torch.gather(do, 0, repeat(indices, 'z -> z d', d=do.shape[1]))
        return dx, None, None


index_put_first_axis = IndexPutFirstAxis.apply


@tensor_cache
def get_unpad_data(
    attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Args:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors.
            `cu_seqlens` shape is [batch_size + 1].
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch


def unpad_input(
    q: torch.Tensor,
    states: Tuple[torch.Tensor],
    attention_mask: torch.Tensor,
    q_len: int,
    keepdim: bool = False,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens
    even though they belong to different batches.


    Arguments:
        q (`torch.Tensor`):
            Query state with padding. Shape: [batch_size, q_len, ...].
        states (`Tuple[torch.Tensor]`):
            Attention state with padding. Shape: [batch_size, seq_len, ...].
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape [batch_size, sequence_length], 1 means valid and 0 means not valid.
        q_len (`int`):
            Target length.
        keepdim (`bool`):
            Whether to keep the batch dimension. Default: `False`.

    Return:
        q (`torch.Tensor`):
            Query state without padding.
            Shape: [1, total_target_length, ...] if `keepdim=True` else [total_target_length, ...].
        states (`Tuple[torch.Tensor]`):
            Attention state without padding.
            Shape: [1, total_source_length, ...] if `keepdim=True` else [total_source_length, ...].
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value),
            used to index into ragged (unpadded) tensors.
            `cu_seqlens` shape is [batch_size + 1].
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence
            i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)
    batch_size, seq_len, *_ = states[0].shape

    state = tuple(
        index_first_axis(rearrange(s, "b s ... -> (b s) ..."), indices_k)
        for s in states
    )

    if q_len == seq_len:
        q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif q_len == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
        indices_q = cu_seqlens_q[:-1]
        q = q.squeeze(1)
    else:
        raise NotImplementedError("We only support either q_len == k_len (prefilling) or q_len == 1 (decoding)")

    if keepdim:
        q = q.unsqueeze(0)
        state = tuple(s.unsqueeze(0) for s in state)

    return (
        q,
        state,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Args:
        hidden_states ([total_tokens, ...]):
            where total_tokens denotes the number of tokens in selected in attention_mask.
        indices ([total_tokens]):
            the indices that represent the non-masked tokens of the original padded input sequence.
        batch_size (int):
            batch_size size for the padded sequence.
        seq_len (int):
            maximum sequence length for the padded sequence.

    Return:
        hidden_states of shape [batch_size, seq_len, ...]
    """
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)
