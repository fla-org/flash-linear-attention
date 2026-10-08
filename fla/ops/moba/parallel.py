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
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward, _flash_attn_varlen_forward
except ImportError:
    flash_attn_varlen_func = None
    _flash_attn_varlen_backward = None
    _flash_attn_varlen_forward = None


def prepare_moba_chunks(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    r"""Split a packed variable-length batch into fixed-size MoBA chunks and emit
    the bookkeeping needed by `parallel_moba`.

    Each sample in `cu_seqlens` is split along the token axis into contiguous
    windows of length `chunk_size`. A chunk never crosses a sample boundary,
    so short samples may produce tail chunks shorter than `chunk_size`. The
    **last chunk of every sample** is excluded from the MoBA *target* pool:
    no future query token can attend to it (causality), and its own tokens
    are still served by the chunk-local self-attn path.

    Args:
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]`, packed-varlen style.
        chunk_size (int):
            MoBA block size.

    Returns:
        A 4-tuple `(cu_chunks, target_chunks, num_target_chunks, chunk_to_batch)`:

        - **cu_chunks** (`torch.Tensor` of shape `[num_chunks + 1]`):
            Cumulative token offsets per chunk (a finer-grained `cu_seqlens`).
        - **target_chunks** (`torch.Tensor` of shape `[num_target_chunks]`):
            Chunk indices that are eligible MoBA targets (all except the last
            chunk of each sample).
        - **num_target_chunks** (`int`):
            Length of `target_chunks`.
        - **chunk_to_batch** (`torch.Tensor` of shape `[num_chunks]`):
            Sample index each chunk belongs to.

    Example:
        Three packed samples of lengths 9, 6, 10 with `chunk_size=4`:

        - Sample 0 → chunks [0,4), [4,8), [8,9)   (tail of length 1)
        - Sample 1 → chunks [9,13), [13,15)        (tail of length 2)
        - Sample 2 → chunks [15,19), [19,23), [23,25)  (tail of length 2)

        Tails (chunks 2, 4, 7) are dropped from the MoBA target pool,
        leaving 5 eligible target chunks.

        >>> cu_seqlens = torch.tensor([0, 9, 15, 25], dtype=torch.int32)
        >>> cu_chunks, target_chunks, n_targets, c2b = prepare_moba_chunks(cu_seqlens, chunk_size=4)
        >>> cu_chunks           # token offsets of all 8 chunks
        tensor([ 0,  4,  8,  9, 13, 15, 19, 23, 25], dtype=torch.int32)
        >>> target_chunks       # MoBA-eligible targets (exclude each sample's tail)
        tensor([0, 1, 3, 5, 6])
        >>> n_targets
        5
        >>> c2b                 # chunk → sample mapping
        tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
    """
    if torch.any(prepare_lens(cu_seqlens) == 0):
        raise ValueError("parallel_moba does not support empty sequences in cu_seqlens")

    # chunk_offsets[b] = first chunk id of sample b
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
    num_chunks = int(chunk_offsets[-1])
    # cu_chunks[c] = start token offset of chunk c (within the packed tensor)
    cu_chunks = prepare_split_cu_seqlens(
        split_size=chunk_size,
        cu_seqlens=cu_seqlens,
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    # chunk_to_batch[c] = sample id of chunk c
    chunk_to_batch = prepare_chunk_indices(cu_seqlens, chunk_size)[:, 0].to(torch.int32)

    # Drop each sample's last chunk from the MoBA target pool (causality).
    is_target = torch.ones(num_chunks, dtype=torch.bool, device=cu_seqlens.device)
    is_target[chunk_offsets[1:] - 1] = False
    target_chunks = is_target.nonzero(as_tuple=True)[0]

    return cu_chunks, target_chunks, len(target_chunks), chunk_to_batch


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
    cu_seqlens: torch.LongTensor,
    max_seqlen: int,
    chunk_size: int,
    topk: int,
) -> torch.Tensor:
    r"""Flash-attn based MoBA implementation.

    MoBA (Mixture of Block Attention, https://arxiv.org/abs/2502.13189) makes
    each query token attend only to a small subset of KV blocks, rather than
    the full causal history. The computation is split into two flash-attn
    passes that are merged via an online-softmax LSE combine:

    1. **Chunking.** Split the packed sequence into fixed-size blocks of length
       `chunk_size` using `prepare_moba_chunks`. Each sample's last chunk is
       reserved for self-attn (it cannot be a MoBA target — causality).
    2. **Gate scoring.** Summarize every candidate block by the mean of its
       keys (one representative key per block), then score every query token
       against every representative via a dot product. Mask out future blocks
       and blocks outside the query's own sample.
    3. **Top-k selection.** For each (query token, head) pair keep the
       `topk - 1` highest-scoring blocks (the local block is always served
       by self-attn, so we only need `topk - 1` extra blocks from MoBA).
    4. **Two-stream attention + online merge.**
       - *Self-attn stream*: chunk-local causal flash-attn within each block.
       - *MoBA stream*: gathered queries attend to their selected blocks'
         full KV via varlen flash-attn.
       - The per-position outputs from the two streams are combined with a
         stable LSE-based online softmax, yielding the same result as a
         single softmax over the union of attended keys.

    Edge cases:
        - If `topk - 1 <= 0` after capping at the number of filtered chunks,
          the op short-circuits to a plain causal `flash_attn_varlen_func`
          over the full sequence (no block selection needed).

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`. When `cu_seqlens` is provided, B
            must be 1 and all sequences are packed along `T`, following the
            FlashAttention varlen convention.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length
            training, consistent with the FlashAttention API.
        max_seqlen (int):
            Max sequence length across the packed batch, consistent with the
            FlashAttention API.
        chunk_size (int):
            Size of each MoBA block.
        topk (int):
            Number of blocks each query attends to, counting the local block.
            With `topk=1` the op degenerates to full causal self-attention.

    Returns:
        torch.Tensor: Output of shape `[B, T, H, V]`, same dtype as `q`.

    Raises:
        ImportError: If `flash-attn` is not installed.
        ValueError: If `q.shape[0] != 1` while `cu_seqlens` is provided.

    Example:
        Single packed sample of 1024 tokens, 4 heads, head-dim 64, chunked
        into 128-token blocks with each query attending to 4 blocks:

        >>> import torch
        >>> from fla.ops.moba import parallel_moba
        >>> B, T, H, D = 1, 1024, 4, 64
        >>> q = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
        >>> k = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
        >>> v = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
        >>> cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device='cuda')
        >>> o = parallel_moba(q, k, v, cu_seqlens, max_seqlen=T, chunk_size=128, topk=4)
        >>> o.shape
        torch.Size([1, 1024, 4, 64])

        Two packed samples of lengths 512 and 256:

        >>> cu_seqlens = torch.tensor([0, 512, 768], dtype=torch.int32, device='cuda')
        >>> q = torch.randn(1, 768, H, D, dtype=torch.float16, device='cuda')
        >>> k = torch.randn(1, 768, H, D, dtype=torch.float16, device='cuda')
        >>> v = torch.randn(1, 768, H, D, dtype=torch.float16, device='cuda')
        >>> o = parallel_moba(q, k, v, cu_seqlens, max_seqlen=512, chunk_size=128, topk=4)
        >>> o.shape
        torch.Size([1, 768, 4, 64])
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
    cu_chunks, target_chunks, num_target_chunks, chunk_to_batch = (
        prepare_moba_chunks(cu_seqlens, chunk_size)
    )

    # the last chunk is always chosen by self-attn, so we only need `topk - 1` from MoBA
    topk = min(topk - 1, num_target_chunks)

    # corner case: no MoBA chunks selectable, fall back to plain causal self-attn
    if topk <= 0:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        ).unsqueeze(0)

    kv = torch.stack((k, v), dim=1)

    self_attn_cu_seqlens = cu_chunks

    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    filtered_kv_indices = torch.arange(
        0, chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_target_chunks, 1)
    filtered_kv_indices += cu_chunks[target_chunks][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    # key_gate_weight [ F_N_CHUNK, H, K ], float32 for better gate logit perception
    key_gate_weight = (
        filtered_kv[:, 0]
        .view(num_target_chunks, chunk_size, H, K)
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
    ].repeat(num_target_chunks, 1)
    chunk_end = cu_chunks[target_chunks + 1]
    batch_end = cu_seqlens[chunk_to_batch[target_chunks] + 1]
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
            num_target_chunks * H + 1 - zero_expert_count,
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
