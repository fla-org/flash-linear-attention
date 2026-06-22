# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

import torch
from einops import repeat

from fla.ops.utils import prepare_chunk_offsets
from fla.ops.utils.pooling import mean_pooling

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = flash_attn_varlen_func = None


def naive_nsa_selection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int = 64,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, TQ, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA is enforced here: the ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >= 16.
            This is a kernel tile dimension, which Triton requires to be a power-of-2 block shape.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, TQ, H, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
        block_size (int):
            Selected block size. Default: `64`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor] or None):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
            When a tuple is provided, it should contain two tensors: `(cu_seqlens_q, cu_seqlens_k)`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, TQ, HQ, V]`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
    q, k, v = map(lambda x: x.float(), (q, k, v))
    B = q.shape[0]

    o = torch.zeros_like(v)
    varlen = True
    if cu_seqlens is None:
        varlen = False
        TQ = TK = q.shape[1]
        cu_q = torch.cat([
            block_indices.new_tensor(range(0, B * TQ, TQ)), block_indices.new_tensor([B * TQ])
        ]).to(device=q.device)
        cu_k = torch.cat([
            block_indices.new_tensor(range(0, B * TK, TK)), block_indices.new_tensor([B * TK])
        ]).to(device=q.device)
    else:
        if isinstance(cu_seqlens, tuple):
            cu_q, cu_k = cu_seqlens
        else:
            cu_q = cu_k = cu_seqlens

    for i in range(len(cu_q) - 1):
        if not varlen:
            q_b, k_b, v_b, i_b = q[i], k[i], v[i], block_indices[i]
        else:
            TQ = cu_q[i+1] - cu_q[i]
            TK = cu_k[i+1] - cu_k[i]
            q_b, k_b, v_b, i_b = (q[0][cu_q[i]:cu_q[i+1]], k[0][cu_k[i]:cu_k[i+1]],
                                  v[0][cu_k[i]:cu_k[i+1]], block_indices[0][cu_q[i]:cu_q[i+1]])
        assert TQ == TK, "TQ != TK case is not supported in naive_nsa_selection"
        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, S*BS, HQ]
        i_b = i_b.view(TQ, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(TQ):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [S*BS, HQ]
            i_i = i_b[i_q]
            # [S*BS, HQ, -1]
            k_i, v_i = map(lambda x: x.gather(0, i_i.clamp(0, TK-1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])),
                           (k_b, v_b))
            # [S*BS, HQ]
            attn = torch.einsum('h d, n h d -> n h', q_i, k_i).masked_fill(
                torch.logical_or(i_i > i_q, i_i < 0), float('-inf')).softmax(0)
            if not varlen:
                o[i, i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)
            else:
                o[0][cu_q[i] + i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)

    return o.to(dtype)


def naive_nsa_compression(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Naive reference for the compressed-attention branch of NSA.

    A query at position `t` attends to compressed block `c` only if the block is fully in the past,
    i.e. `(c + 1) * block_size - 1 <= t`.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, TQ, HQ, K]`.
        k_cmp (torch.Tensor):
            Compressed keys of shape `[B, TC, H, K]`, where `TC` is the number of compressed blocks.
        v_cmp (torch.Tensor):
            Compressed values of shape `[B, TC, H, V]`.
        block_size (int):
            Compression block size.
        scale (float):
            Scale factor for attention scores.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs. Default: `None`.

    Returns:
        o (torch.Tensor):
            Compressed-attention outputs of shape `[B, TQ, HQ, V]`.
        lse (torch.Tensor):
            Log-sum-exp of attention scores of shape `[B, TQ, HQ]`, `-inf` where no block is visible yet.
    """
    dtype = q.dtype
    H = k_cmp.shape[2]
    G = q.shape[2] // H
    q, k_cmp, v_cmp = (x.float() for x in (q, k_cmp, v_cmp))
    k_cmp, v_cmp = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k_cmp, v_cmp))

    def attend(q_b, k_b, v_b):
        # q_b: [TQ, HQ, K], k_b/v_b: [TC, HQ, *]
        TQ, TC = q_b.shape[0], k_b.shape[0]
        attn = torch.einsum('t h d, c h d -> t h c', q_b, k_b) * scale
        i_t = torch.arange(TQ, device=q_b.device)[:, None]
        i_c = torch.arange(TC, device=q_b.device)[None, :]
        allow = ((i_c + 1) * block_size - 1) <= i_t
        attn = attn.masked_fill(~allow[:, None, :], float('-inf'))
        lse_b = torch.logsumexp(attn, dim=-1)
        # queries with no visible block softmax to nan -> 0
        attn = torch.nan_to_num(torch.softmax(attn, dim=-1), nan=0.0)
        o_b = torch.einsum('t h c, c h v -> t h v', attn, v_b)
        return o_b, lse_b

    if cu_seqlens is None:
        o, lse = zip(*(attend(q[i], k_cmp[i], v_cmp[i]) for i in range(q.shape[0])))
        o, lse = torch.stack(o, 0), torch.stack(lse, 0)
    else:
        cu_q, cu_k = cu_seqlens, prepare_chunk_offsets(cu_seqlens, block_size)
        o, lse = zip(*(
            attend(q[0, cu_q[i]:cu_q[i + 1]], k_cmp[0, cu_k[i]:cu_k[i + 1]], v_cmp[0, cu_k[i]:cu_k[i + 1]])
            for i in range(len(cu_q) - 1)
        ))
        o, lse = torch.cat(o, 0).unsqueeze(0), torch.cat(lse, 0).unsqueeze(0)
    return o.to(dtype), lse


def naive_nsa_topk(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    block_counts: int | torch.Tensor,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor] | None = None,
) -> torch.LongTensor:
    r"""
    Naive reference for NSA top-k block selection.

    For each query, blocks are ranked by their attention probability averaged over the query group,
    and the top `block_counts` blocks are kept. The first block and the current/previous blocks are
    always selected; causally-invisible or surplus slots are padded with `-1`.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, TQ, HQ, K]`.
        k_cmp (torch.Tensor):
            Compressed keys of shape `[B, TC, H, K]`, where `TC` is the number of compressed blocks.
        block_counts (int or torch.Tensor):
            Number of blocks to select per query. Either an int, or a tensor of shape `[B, TQ, H]`.
        block_size (int):
            Compression block size.
        scale (float):
            Scale factor for attention scores.
        cu_seqlens (torch.LongTensor, tuple or None, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs.
            A tuple holds `(cu_seqlens_q, cu_seqlens_k)`. Default: `None`.

    Returns:
        block_indices (torch.LongTensor):
            Selected block indices of shape `[B, TQ, H, S]`, padded with `-1`.
    """
    B, TQ, HQ, _ = q.shape
    H = k_cmp.shape[2]
    G = HQ // H
    k_cmp = repeat(k_cmp, 'b t h d -> b t (h g) d', g=G)

    device = q.device
    varlen = True
    if cu_seqlens is None:
        varlen = False
        TQ = q.shape[1]
        TC = k_cmp.shape[1]
        cu_q = torch.cat([torch.arange(0, B * TQ, TQ), torch.tensor([B * TQ])])
        cu_k = torch.cat([torch.arange(0, B * TC, TC), torch.tensor([B * TC])])
    else:
        assert B == 1
        if isinstance(cu_seqlens, tuple):
            cu_q, cu_k = cu_seqlens
        else:
            cu_q = cu_k = cu_seqlens
        cu_k = prepare_chunk_offsets(cu_k, block_size)

    if isinstance(block_counts, int):
        S = int(block_counts)
        assert S >= 0, "block_counts (int) must be >= 0"
    elif torch.is_tensor(block_counts):
        S = int(block_counts.max().item())
    block_indices = torch.full((B, TQ, H, S), -1, device=device, dtype=torch.long)

    for i in range(len(cu_q) - 1):
        if not varlen:
            q_b, k_b = q[i], k_cmp[i]
        else:
            TQ = (cu_q[i+1] - cu_q[i]).item()
            TC = (cu_k[i+1] - cu_k[i]).item()
            q_b, k_b = q[0][cu_q[i]:cu_q[i+1]], k_cmp[0][cu_k[i]:cu_k[i+1]]

        # [TQ, H, G, TC]
        attn = torch.einsum('t h d, c h d -> t h c', q_b, k_b).reshape(TQ, H, G, TC) * scale
        i_t = torch.arange(TQ, device=device).unsqueeze(1)
        i_c = torch.arange(TC, device=device).unsqueeze(0)
        # block c is causally visible once its last token (c + 1) * block_size - 1 is in the past
        allow_causal = ((i_c + 1) * block_size - 1) <= i_t  # [TQ, TC]
        # the first block and the query's own/previous block are always selected
        i_blk = i_t // block_size
        forced = (i_c == i_blk) | (i_c == 0) | (i_c == i_blk - 1)  # [TQ, TC]
        attn = attn.masked_fill(~allow_causal[:, None, None, :], float('-inf'))
        allow = allow_causal | forced

        probs = torch.nan_to_num(torch.softmax(attn, dim=-1), nan=0.0)  # [TQ, H, G, TC]
        scores = probs.mean(dim=2)  # [TQ, H, TC]
        scores = torch.where(forced[:, None, :], 1.0, scores)

        if isinstance(block_counts, int):
            n_sel = torch.full((TQ, H), S, dtype=torch.long, device=device)
        elif torch.is_tensor(block_counts):
            if varlen:
                assert block_counts.shape == (1, TQ, H)
                n_sel = block_counts[0].to(device=device, dtype=torch.long)
            else:
                assert block_counts.shape == (B, TQ, H)
                n_sel = block_counts[i].to(device=device, dtype=torch.long)
        else:
            raise TypeError("block_counts must be int or torch.Tensor")

        _, i_top = torch.topk(scores, k=min(S, TC), dim=-1)  # [TQ, H, min(S, TC)]

        # keep a selected block only if it is allowed and within the per-query quota, else pad with -1
        top_allowed = torch.gather(allow[:, None, :].expand(TQ, H, TC).long(), dim=-1, index=i_top).bool()
        i_s = torch.arange(S, device=device).view(1, 1, S)
        in_quota = (i_s < n_sel.unsqueeze(-1))[:, :, :TC]
        sel = torch.where(top_allowed & in_quota, i_top, torch.full_like(i_top, -1))

        if S > TC:
            sel = torch.cat((sel, torch.full((TQ, H, S - TC), -1, device=device, dtype=i_top.dtype)), dim=-1)
        if varlen:
            block_indices[0, cu_q[i]:cu_q[i+1]] = sel
        else:
            block_indices[i] = sel
    return block_indices


def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor | None = None,
    g_slc: torch.Tensor | None = None,
    g_swa: torch.Tensor | None = None,
    block_indices: torch.LongTensor | None = None,
    block_counts: torch.LongTensor | int = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor] | None = None,
    return_block_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.LongTensor]:
    r"""
    Naive reference for NSA, combining the compressed, selected, and sliding-window branches.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, TQ, HQ, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
            GQA is enforced here: the ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >= 16.
            This is a kernel tile dimension, which Triton requires to be a power-of-2 block shape.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g_cmp (torch.Tensor, Optional):
            Gate score for compressed attention of shape `[B, TQ, HQ]`. Default: `None`.
        g_slc (torch.Tensor, Optional):
            Gate score for selected attention of shape `[B, TQ, HQ]`. Default: `None`.
        g_swa (torch.Tensor, Optional):
            Gate score for sliding-window attention of shape `[B, TQ, HQ]`. Default: `None`.
        block_indices (torch.LongTensor, Optional):
            Block indices of shape `[B, TQ, H, S]`, where `S` is the number of selected blocks per query.
            If provided, overrides the selection computed from compression when `g_cmp` is given. Default: `None`.
        block_counts (torch.LongTensor or int, Optional):
            Number of selected blocks per query. A tensor of shape `[B, TQ, H]`, or an int. Default: 16.
        block_size (int, Optional):
            Selected block size. Default: 64.
        window_size (int, Optional):
            Sliding window size. Default: 0.
        scale (float, Optional):
            Scale factor for attention scores. If `None`, defaults to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor, tuple or None, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs, consistent with the
            FlashAttention API. A tuple holds `(cu_seqlens_q, cu_seqlens_k)`. Default: `None`.
        return_block_indices (bool, Optional):
            Whether to also return the selected block indices. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, TQ, HQ, V]`. When `return_block_indices=True`, also returns the
            selected block indices of shape `[B, TQ, H, S]`.
    """
    assert block_counts is not None, "block counts must be provided for selection"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    G = q.shape[2] // k.shape[2]
    assert G >= 16 and (G & (G - 1)) == 0, "Group size (HQ/H) must be a power of 2 and >= 16 in NSA"

    if cu_seqlens is not None:
        if isinstance(cu_seqlens, tuple):
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
        else:
            cu_seqlens_q = cu_seqlens_k = cu_seqlens
    else:
        cu_seqlens_q = cu_seqlens_k = None

    k_cmp, v_cmp = mean_pooling(k, block_size, cu_seqlens), mean_pooling(v, block_size, cu_seqlens)
    o_cmp = None
    if g_cmp is not None:
        o_cmp, _ = naive_nsa_compression(
            q=q,
            k_cmp=k_cmp,
            v_cmp=v_cmp,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens
        )
        if block_indices is None:
            block_indices = naive_nsa_topk(
                q=q,
                k_cmp=k_cmp,
                block_counts=block_counts,
                block_size=block_size,
                scale=scale,
                cu_seqlens=cu_seqlens
            )
        else:
            warnings.warn("`block_indices` is provided, overriding the selection computed from compression")
    o = o_slc = naive_nsa_selection(q, k, v, block_indices, block_size, scale, cu_seqlens)
    if g_slc is not None:
        o = o_slc * g_slc.unsqueeze(-1)
    if o_cmp is not None:
        o = torch.addcmul(o, o_cmp, g_cmp.unsqueeze(-1))
    if window_size > 0:
        if cu_seqlens is not None:
            o_swa = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                causal=True,
                window_size=(window_size-1, 0)
            ).unsqueeze(0)
        else:
            o_swa = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(window_size-1, 0)
            )
        o = torch.addcmul(o, o_swa, g_swa.unsqueeze(-1))
    if return_block_indices:
        return o, block_indices
    else:
        return o
