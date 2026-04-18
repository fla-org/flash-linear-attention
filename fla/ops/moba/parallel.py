# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from einops import rearrange

from fla.ops.utils.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_cu_seqlens_from_lens,
    prepare_lens,
    prepare_split_cu_seqlens,
)

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import (
        _flash_attn_varlen_backward,
        _flash_attn_varlen_forward,
    )
except ImportError:
    flash_attn_varlen_func = None
    _flash_attn_varlen_backward = None
    _flash_attn_varlen_forward = None


def calc_chunks(cu_seqlens, chunk_size):
    """calc chunks that needs moba attention"""
    if torch.any(prepare_lens(cu_seqlens) == 0):
        raise ValueError("parallel_moba does not support empty sequences in cu_seqlens")

    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = prepare_chunk_offsets(cu_seqlens, chunk_size)
    num_chunk = int(cu_num_chunk[-1])
    # cu_chunk[chunk_idx] = start token offset of chunk idx (within packed tensor)
    cu_chunk = prepare_split_cu_seqlens(
        batch_size=0,
        seq_len=0,
        split_size=chunk_size,
        cu_seqlens=cu_seqlens,
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = prepare_chunk_indices(cu_seqlens, chunk_size)[:, 0].to(torch.int32)

    # filter chunks (remove last chunk of each batch)
    chunk_to_remain = torch.ones(num_chunk, dtype=torch.bool, device=cu_seqlens.device)
    chunk_to_remain[cu_num_chunk[1:] - 1] = False
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]

    return cu_chunk, filtered_chunk_indices, len(filtered_chunk_indices), chunk_to_batch


class ParallelMoBAFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlens,
        moba_q,
        moba_kv,
        moba_cu_seqlens_q,
        moba_cu_seqlens_k,
        max_seqlen,
        chunk_size,
        moba_q_sh_indices,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.chunk_size = chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # self attn
        self_attn_out_sh, self_attn_lse_hs, _, _ = (
            _flash_attn_varlen_forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self_attn_cu_seqlens,
                cu_seqlens_k=self_attn_cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=True,
                dropout_p=0.0,
            )
        )

        # moba attn
        moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlens_q,
            cu_seqlens_k=moba_cu_seqlens_k,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )

        # convert lse shape hs -> sh ( follow the legacy mix attn logic )
        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        # output buffer [T, H, K], same shape as q
        output = torch.zeros_like(q, dtype=torch.float32)

        # flatten T & H for index ops
        output_2d = output.view(-1, q.shape[-1])

        # calc mixed_lse
        # minus max lse to avoid exp explosion
        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
            .reshape_as(moba_attn_lse)
        )

        mixed_attn_se_sh = self_attn_lse_sh.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se_sh.view(-1).index_add_(
            0, moba_q_sh_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse_sh = mixed_attn_se_sh.log()

        # add attn output
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ T, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        # add moba output
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ T, H ]
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
        output = output.to(q.dtype)
        # add back max lse
        mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)
        ctx.save_for_backward(
            output,
            mixed_attn_lse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlens,
            moba_q,
            moba_kv,
            moba_cu_seqlens_q,
            moba_cu_seqlens_k,
            moba_q_sh_indices,
        )

        return output

    @staticmethod
    def backward(ctx, d_output):
        max_seqlen, chunk_size, softmax_scale = ctx.max_seqlen, ctx.chunk_size, ctx.softmax_scale
        (
            output, mixed_attn_vlse_sh, q, k, v, self_attn_cu_seqlens, moba_q,
            moba_kv, moba_cu_seqlens_q, moba_cu_seqlens_k, moba_q_sh_indices,
        ) = ctx.saved_tensors
        d_output = d_output.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        _ = _flash_attn_varlen_backward(
            dout=d_output,
            q=q,
            k=k,
            v=v,
            out=output,
            softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=self_attn_cu_seqlens,
            cu_seqlens_k=self_attn_cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
            dropout_p=0.0,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        K = q.shape[-1]
        d_moba_output = d_output.view(-1, K).index_select(0, moba_q_sh_indices).unsqueeze(1)
        moba_output = output.view(-1, K).index_select(0, moba_q_sh_indices).unsqueeze(1)
        mixed_attn_vlse = mixed_attn_vlse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)

        dmq = torch.empty_like(moba_q)
        dmk = torch.empty_like(moba_kv[:, 0])
        dmv = torch.empty_like(moba_kv[:, 1])

        _ = _flash_attn_varlen_backward(
            dout=d_moba_output,
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=dmq,
            dk=dmk,
            dv=dmv,
            cu_seqlens_q=moba_cu_seqlens_q,
            cu_seqlens_k=moba_cu_seqlens_k,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        dmkv = torch.stack((dmk, dmv), dim=1)
        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None


def parallel_moba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    chunk_size: int,
    topk: int,
) -> torch.Tensor:
    r"""Flash-attn based MoBA implementation. Core logic:
    1. Calculate the chunks and the number of chunks, n = floor(data_size / chunk_size)
       - tokens in the tail chunk are reserved for self attn
       - tokens in other chunks will be processed in later steps
    2. K in each chunk will calculate mean value as the representative k, and Q will attend to these representative
    k to get the gate logit, which will be used to select topk chunks
    3. Select the topk chunks and get the dense q for each kv chunk pair and do the varlen attention
    4. Combine the varlen attn and self attn results via online softmax to get the final result

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`. When `cu_seqlens` is provided, B must
            be 1 and all sequences are packed along `T`, following the FlashAttention
            varlen convention.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        max_seqlen (int):
            Max sequence length across the batch, consistent with the FlashAttention API.
        chunk_size (int):
            Size of each MoBA chunk.
        topk (int):
            Number of chunks each query attends to (including its own current chunk).

    Returns:
        Output of shape `[B, T, H, V]`.
    """
    if flash_attn_varlen_func is None:
        raise ImportError(
            "`parallel_moba` requires `flash-attn`. Install it via `pip install flash-attn`."
        )
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )

    # The underlying `_flash_attn_varlen_forward/backward` kernels expect packed
    # 3-D `[total_T, H, D]`; squeeze the leading batch dim here and restore it
    # on the output.
    q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
    T, H, K = q.shape

    # prepare chunk meta
    cu_chunk, filtered_chunk_indices, num_filtered_chunk, chunk_to_batch = (
        calc_chunks(cu_seqlens, chunk_size)
    )

    # the last chunk is always chosen by self-attn, so we only need `topk - 1` from MoBA
    topk = min(topk - 1, num_filtered_chunk)

    # corner case: no MoBA chunks selectable, fall back to plain causal self-attn
    if topk <= 0:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        ).unsqueeze(0)

    kv = torch.stack((k, v), dim=1)

    self_attn_cu_seqlens = cu_chunk

    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    filtered_kv_indices = torch.arange(
        0, chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    # key_gate_weight [ F_N_CHUNK, H, K ], float32 for better gate logit perception
    key_gate_weight = (
        filtered_kv[:, 0]
        .view(num_filtered_chunk, chunk_size, H, K)
        .mean(dim=1)
        .float()
    )
    q = q.float()
    # gate [ F_N_CHUNK, H, T ]
    gate = torch.einsum("nhk,thk->nht", key_gate_weight, q)
    key_gate_weight = key_gate_weight.type_as(k)
    q = q.type_as(k)

    # mask out chunks that lie outside the current sequence, and the current chunk itself
    gate_seq_idx = torch.arange(0, T, device=q.device, dtype=torch.int32)[
        None, :
    ].repeat(num_filtered_chunk, 1)
    chunk_end = cu_chunk[filtered_chunk_indices + 1]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    gate_chunk_end_mask = gate_seq_idx < chunk_end[:, None]
    gate_batch_end_mask = gate_seq_idx >= batch_end[:, None]
    gate_inf_mask = gate_chunk_end_mask | gate_batch_end_mask
    gate.masked_fill_(gate_inf_mask.unsqueeze(1), -float("inf"))

    # find topk chunks per (head, token), then AND with the causal mask
    # gate_mask [ N_CHUNK, H, T ], True means the (chunk, head, token) triple participates in MoBA attn
    _, gate_top_k_idx = torch.topk(gate, k=topk, dim=0, largest=True, sorted=False)
    gate_mask = torch.logical_not(gate.isinf())
    gate_idx_mask = torch.zeros_like(gate_mask).scatter_(dim=0, index=gate_top_k_idx, value=True)
    gate_mask = torch.logical_and(gate_mask, gate_idx_mask)

    # varlen trick: combining all q index that needs moba attn
    # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[-1]  # [HT] * N
    # moba_seqlens_q[i]: number of q tokens selected for the i-th kv (chunk, head) pair
    moba_seqlens_q = gate_mask.sum(dim=-1).flatten()
    # gather the selected q tokens, shape [ selected_T, K ]
    moba_q = rearrange(q, "t h k -> (h t) k").index_select(0, moba_q_indices)
    moba_q = moba_q.unsqueeze(1)
    # moba_q_sh_indices: position of each gathered q token inside the original q tensor
    moba_q_sh_indices = moba_q_indices % T * H + moba_q_indices // T

    # reorganize kv to align with moba_q (grouped as (H, chunk) pairs)

    # cut off (chunk, head) pairs whose q selection is empty
    q_zero_mask = moba_seqlens_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    if zero_expert_count > 0:
        moba_seqlens_q = moba_seqlens_q[valid_expert_mask]
    # moba cu_seqlens for flash-attn varlen
    moba_cu_seqlens_q = prepare_cu_seqlens_from_lens(moba_seqlens_q)
    moba_kv = rearrange(filtered_kv, "t x h k -> h t x k")
    moba_kv = moba_kv.split(chunk_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0)
    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        # drop (chunk, head) pairs with zero q, otherwise grads may be nan
        moba_kv = moba_kv[valid_expert_mask]
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)
    moba_cu_seqlens_k = (
        torch.arange(
            0,
            num_filtered_chunk * H + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * chunk_size
    )

    assert moba_cu_seqlens_k.shape == moba_cu_seqlens_q.shape, (
        f"moba_cu_seqlens_k.shape != moba_cu_seqlens_q.shape "
        f"{moba_cu_seqlens_k.shape} != {moba_cu_seqlens_q.shape}"
    )

    return ParallelMoBAFunction.apply(
        q,
        k,
        v,
        self_attn_cu_seqlens,
        moba_q,
        moba_kv,
        moba_cu_seqlens_q,
        moba_cu_seqlens_k,
        max_seqlen,
        chunk_size,
        moba_q_sh_indices,
    ).unsqueeze(0)
