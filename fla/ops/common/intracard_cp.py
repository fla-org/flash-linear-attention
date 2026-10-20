# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Intra-card context parallelism for variable-length delta-rule inference.

Long sequences are split at chunk boundaries. A pre-scan summarizes each subsequence as an affine state transform;
merging those summaries supplies the initial states needed to run the subsequences independently.
Partition indices and device tensors are cached together to avoid repeating host work and device transfers.
"""

from __future__ import annotations

import logging
import weakref
from collections import OrderedDict
from typing import NamedTuple

import torch
import triton

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from fla.ops.cp.chunk_delta_h import pre_process_fwd_kernel_merged
from fla.ops.cp.context import _check_cp_device, _validate_cp_kwargs
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import get_multiprocessor_count

logger = logging.getLogger(__name__)


# cache indices and device tensors by cu_seqlens identity
_intracard_cache: OrderedDict[tuple, _CacheEntry] = OrderedDict()
_INTRACARD_CACHE_MAXSIZE = 32


class _CacheEntry(NamedTuple):
    """Cached partition metadata and device tensors for one sequence-boundary object."""
    # a weak reference prevents stale cache hits when Python reuses an object id
    cu_seqlens_ref: weakref.ReferenceType[torch.Tensor]
    cu_seqlens_subseq_values: list[int]
    split_info: SplitSeqInfo
    total_subseqs: int
    cu_seqlens_split_values: list[int]
    S_split_total: int
    non_first_indices: list[int]
    first_subseq_indices: list[int]
    last_subseq_indices: list[int]
    num_non_first: int
    merge_seq_offsets: list[int]
    merge_init_offsets: list[int]
    # cached device tensors avoid repeated host-to-device transfers
    cu_seqlens_subseq_gpu: torch.Tensor
    cu_seqlens_split_flat: torch.Tensor


class SplitSeqInfo(NamedTuple):
    """Original sequence ids and their contiguous subsequence ranges."""
    split_seq_ids: list[int]       # [num_split_seqs] original sequence indices
    start_subseq_idx: list[int]    # [num_split_seqs] start index in subseq array
    num_subseqs: list[int]         # [num_split_seqs] number of sub-sequences per split

    @property
    def num_split_seqs(self) -> int:
        return len(self.split_seq_ids)

    def __bool__(self) -> bool:
        return self.num_split_seqs > 0


def _raw_chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta['BV']) * N * HV, )

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        HV=HV,
        H=H,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return h, v_new, final_state


def compute_subseq_len(
    seq_len: int,
    num_sms: int,
    num_heads: int,
    chunk_size: int = 64,
) -> int:
    """Choose chunk-aligned subsequences that balance SM occupancy and merge overhead.

    The longest sequence determines the split target.
    A minimum subsequence size avoids excessive splitting and merge overhead for shorter sequences in mixed-length batches.
    """
    seq_chunks = (seq_len + chunk_size - 1) // chunk_size

    if seq_chunks < 8:
        return seq_len

    # each subsequence contributes NUM_V_BLOCKS * num_heads blocks
    NUM_V_BLOCKS = 2
    target_splits = max(4, num_sms // (NUM_V_BLOCKS * num_heads))

    subseq_chunks = (seq_chunks + target_splits - 1) // target_splits

    # keep short sequences unsplit to amortize pre-scan and merge overhead
    MIN_SUBSEQ_CHUNKS = 128
    subseq_chunks = max(subseq_chunks, MIN_SUBSEQ_CHUNKS)

    return subseq_chunks * chunk_size


def prepare_subseq_cu_seqlens(
    cu_seqlens_cpu: torch.Tensor,
    subseq_len: int,
    chunk_size: int = 64,
    max_splits: int = 32,
) -> tuple[list[int], SplitSeqInfo | bool, int]:
    """Insert chunk-aligned subsequence boundaries for long sequences.

    Returns:
        boundaries (list[int]):
            Cumulative subsequence boundaries, including unsplit sequences.
        split_info (SplitSeqInfo | bool):
            Split sequence metadata, or `False` when no sequence needs splitting.
        total_subseqs (int):
            Number of subsequences, or zero when no sequence needs splitting.
    """
    N = len(cu_seqlens_cpu) - 1
    if N == 0:
        return cu_seqlens_cpu.tolist(), False, 0

    subseq_chunks = (subseq_len + chunk_size - 1) // chunk_size
    # splitting needs enough work to amortize the extra pre-scan and merge
    threshold_subseq_len = 3 * subseq_len

    split_seq_ids: list[int] = []
    start_subseq_idxs: list[int] = []
    num_subseqs_list: list[int] = []

    boundaries: list[int] = [0]
    cumsum_offset = 0

    for i in range(N):
        seq_start = int(cu_seqlens_cpu[i].item())
        seq_end = int(cu_seqlens_cpu[i + 1].item())
        seq_len_i = seq_end - seq_start
        seq_chunks_i = (seq_len_i + chunk_size - 1) // chunk_size

        if seq_len_i >= threshold_subseq_len:
            num_ss = min(max_splits, (seq_chunks_i + subseq_chunks - 1) // subseq_chunks)
            chunks_per = (seq_chunks_i + num_ss - 1) // num_ss
            actual_ssl = chunks_per * chunk_size

            # preserve chunk alignment inside the sequence; only its final subsequence may end in a partial chunk
            split_seq_ids.append(i)
            start_subseq_idxs.append(cumsum_offset)
            num_subseqs_list.append(num_ss)

            for j in range(num_ss):
                boundary = min(seq_start + (j + 1) * actual_ssl, seq_end)
                boundaries.append(boundary)
            cumsum_offset += num_ss
        else:
            boundaries.append(seq_end)
            cumsum_offset += 1

    if not split_seq_ids:
        return cu_seqlens_cpu.tolist(), False, 0

    total_subseqs = cumsum_offset

    split_info = SplitSeqInfo(split_seq_ids=split_seq_ids, start_subseq_idx=start_subseq_idxs, num_subseqs=num_subseqs_list)

    return boundaries, split_info, total_subseqs


def intracard_pre_scan(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    cu_seqlens_subseq_split: torch.Tensor,
    S_split: int,
    chunk_size: int = 64,
    **kwargs,
):
    _validate_cp_kwargs(**kwargs)
    _check_cp_device(kg.device)
    H, K, V, HV = kg.shape[2], kg.shape[3], u.shape[3], u.shape[2]
    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64

    hm = kg.new_empty(S_split, HV, K, V + K, dtype=torch.float32)

    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV, S_split)
    pre_process_fwd_kernel_merged[grid](
        k=kg,
        v=u,
        w=w,
        g=g,
        gk=gk,
        bg=None,
        u=u,
        hm=hm,
        cu_seqlens=cu_seqlens_subseq_split,
        T=0,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BK1=BK,
        MULTI_SEQS=True,
    )

    return hm


def intracard_merge(
    hm: torch.Tensor,
    split_info: SplitSeqInfo,
    num_non_first: int,
    merge_seq_offsets: list[int],
    merge_init_offsets: list[int],
    device: torch.device,
    initial_state: torch.Tensor | None = None,
    state_v_first: bool = False,
    **kwargs,
) -> tuple[torch.Tensor | None, int]:
    """Compute initial states for non-first subsequences from their affine summaries.

    Each original sequence starts from its supplied initial_state, or zero.
    Returned states are ordered by split sequence, with the first subsequence of each sequence omitted.
    """
    from fla.ops.cp.chunk_delta_h import merge_fwd_bwd_kernel

    _validate_cp_kwargs(**kwargs)
    _check_cp_device(hm.device)
    if num_non_first == 0:
        return None, 0

    HV = hm.shape[1]
    K = hm.shape[2]
    V = hm.shape[3] - K
    BK = triton.next_power_of_2(K)

    num_split_seqs = split_info.num_split_seqs

    # combine metadata into one device transfer to reduce synchronization
    all_int_data = merge_seq_offsets + merge_init_offsets + split_info.split_seq_ids
    all_tensor = torch.tensor(all_int_data, dtype=torch.int32, device=device)
    n_so = len(merge_seq_offsets)
    n_io = len(merge_init_offsets)
    seq_offsets = all_tensor[:n_so]
    init_offsets = all_tensor[n_so:n_so+n_io]
    h0_seq_ids = all_tensor[n_so+n_io:]

    if state_v_first:
        initial_states_merge = hm.new_empty(num_non_first, HV, V, K, dtype=torch.float32)
    else:
        initial_states_merge = hm.new_empty(num_non_first, HV, K, V, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), num_split_seqs, HV)

    merge_fwd_bwd_kernel[grid](
        h=initial_states_merge,
        ag_hm=hm,
        pre_or_post_num_ranks=num_split_seqs,
        rank=0,
        seq_offsets=seq_offsets,
        init_offsets=init_offsets,
        h0_seq_ids=h0_seq_ids,
        h0=initial_state,
        h_seq_idx=None,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        FORWARD=True,
        INTRACARD_MODE=True,
        NUM_SEQ_ENTRIES=num_split_seqs,
        STATE_V_FIRST=state_v_first,
    )

    return initial_states_merge, num_non_first


def _precompute_intracard_indices(
    split_info: SplitSeqInfo,
    cu_seqlens_subseq_values: list[int],
    N_orig: int,
) -> tuple[list[int], int, list[int], list[int], list[int], int, list[int], list[int]]:
    """Prepare sequence indices for pre-scan, state merging, and final-state gathering.

    Returns:
        cu_seqlens_split_values (list[int]):
            Flattened boundaries of split sequences for pre-scan.
        S_split_total (int):
            Total subsequences belonging to split sequences.
        non_first_indices (list[int]):
            Destinations for merged initial states.
        first_subseq_indices (list[int]):
            Destinations for original initial states.
        last_subseq_indices (list[int]):
            Sources for gathering final states.
        num_non_first (int):
            Number of initial states produced by the merge.
        merge_seq_offsets (list[int]):
            Cumulative subsequence counts for the merge kernel.
        merge_init_offsets (list[int]):
            Cumulative non-first subsequence counts for the merge kernel.
    """
    starts = split_info.start_subseq_idx
    num_ss = split_info.num_subseqs
    split_ids = split_info.split_seq_ids

    # cu_seqlens_split_values: for each split seq, extract [start:start+n+1] boundaries
    cu_seqlens_split_values: list[int] = []
    S_split_total = 0
    for s, n in zip(starts, num_ss):
        cu_seqlens_split_values.extend(cu_seqlens_subseq_values[s:s+n+1])
        S_split_total += n

    # num_subseqs_per_seq: [N_orig], default 1 for unsplit sequences
    num_subseqs_per_seq = [1] * N_orig
    for sid, nss in zip(split_ids, num_ss):
        num_subseqs_per_seq[sid] = nss

    # non_first_indices: for scattering merged initial states
    non_first_indices: list[int] = []
    for s, n in zip(starts, num_ss):
        for j in range(1, n):
            non_first_indices.append(s + j)

    # first_subseq_indices: for scattering original initial states
    first_subseq_indices: list[int] = [0]
    running = 0
    for i in range(N_orig - 1):
        running += num_subseqs_per_seq[i]
        first_subseq_indices.append(running)

    # last_subseq_indices: for gathering final states
    last_subseq_indices: list[int] = []
    running = 0
    for n in num_subseqs_per_seq:
        running += n
        last_subseq_indices.append(running - 1)

    merge_seq_offsets: list[int] = [0]
    merge_init_offsets: list[int] = [0]
    for n in num_ss:
        merge_seq_offsets.append(merge_seq_offsets[-1] + n)
        merge_init_offsets.append(merge_init_offsets[-1] + n - 1)
    num_non_first = merge_init_offsets[-1]

    return (
        cu_seqlens_split_values,
        S_split_total,
        non_first_indices,
        first_subseq_indices,
        last_subseq_indices,
        num_non_first,
        merge_seq_offsets,
        merge_init_offsets,
    )


def intracard_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    max_splits: int = 32,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run independent subsequences after reconstructing their incoming recurrent states.

    The original initial state belongs to each sequence's first subsequence; later ones use affine-merge results.
    Reusing a cu_seqlens object assumes its boundaries are unchanged while its cache entry is retained.
    """
    _validate_cp_kwargs(**kwargs)
    _check_cp_device(k.device)
    assert cu_seqlens is not None, "intracard_fwd_h requires cu_seqlens"

    _, _, _Hq, K = k.shape
    V = u.shape[-1]
    HV = u.shape[2]
    device = k.device

    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()

    seq_lens = torch.diff(cu_seqlens_cpu)
    max_seq_len = int(seq_lens.max().item())
    num_sms = get_multiprocessor_count()
    subseq_len = compute_subseq_len(max_seq_len, num_sms, HV, chunk_size)

    early_return = (seq_lens < 2 * subseq_len).all()

    cached = None
    cache_key = None

    if not early_return:
        # identity avoids copying device boundaries to form a content key; the weakref guards against id reuse
        cache_key = (
            id(cu_seqlens),
            subseq_len,
            chunk_size,
            max_splits,
            str(device),
        )
        cached = _intracard_cache.get(cache_key)
        if cached is not None:
            # an id can be reused after the original tensor is collected
            if cached.cu_seqlens_ref() is cu_seqlens:
                _intracard_cache.move_to_end(cache_key)
            else:
                _intracard_cache.pop(cache_key, None)
                cached = None

        if cached is not None:
            cu_seqlens_subseq_values = cached.cu_seqlens_subseq_values
            split_info = cached.split_info
            total_subseqs = cached.total_subseqs
            cu_seqlens_split_values = cached.cu_seqlens_split_values
            S_split_total = cached.S_split_total
            non_first_indices = cached.non_first_indices
            first_subseq_indices = cached.first_subseq_indices
            last_subseq_indices = cached.last_subseq_indices
            num_non_first = cached.num_non_first
            merge_seq_offsets = cached.merge_seq_offsets
            merge_init_offsets = cached.merge_init_offsets
            cu_seqlens_subseq_gpu = cached.cu_seqlens_subseq_gpu
            cu_seqlens_split_flat = cached.cu_seqlens_split_flat
        else:
            cu_seqlens_subseq_values, split_info, total_subseqs = prepare_subseq_cu_seqlens(
                cu_seqlens_cpu,
                subseq_len,
                chunk_size,
                max_splits=max_splits,
            )

    if early_return or not split_info:
        return _raw_chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )

    N_orig = len(cu_seqlens_cpu) - 1

    if cached is None:
        (
            cu_seqlens_split_values,
            S_split_total,
            non_first_indices,
            first_subseq_indices,
            last_subseq_indices,
            num_non_first,
            merge_seq_offsets,
            merge_init_offsets,
        ) = _precompute_intracard_indices(split_info, cu_seqlens_subseq_values, N_orig)

        dtype = cu_seqlens_cpu.dtype
        cu_seqlens_subseq_gpu = torch.tensor(cu_seqlens_subseq_values, dtype=dtype, device=device)
        cu_seqlens_split_flat = torch.tensor(cu_seqlens_split_values, dtype=dtype, device=device)

        _intracard_cache[cache_key] = _CacheEntry(
            cu_seqlens_ref=weakref.ref(cu_seqlens),
            cu_seqlens_subseq_values=cu_seqlens_subseq_values,
            split_info=split_info,
            total_subseqs=total_subseqs,
            cu_seqlens_split_values=cu_seqlens_split_values,
            S_split_total=S_split_total,
            non_first_indices=non_first_indices,
            first_subseq_indices=first_subseq_indices,
            last_subseq_indices=last_subseq_indices,
            num_non_first=num_non_first,
            merge_seq_offsets=merge_seq_offsets,
            merge_init_offsets=merge_init_offsets,
            cu_seqlens_subseq_gpu=cu_seqlens_subseq_gpu,
            cu_seqlens_split_flat=cu_seqlens_split_flat,
        )
        while len(_intracard_cache) > _INTRACARD_CACHE_MAXSIZE:
            _intracard_cache.popitem(last=False)

    # summarize only split sequences; unsplit sequences keep their original initial states
    hm = intracard_pre_scan(
        kg=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        cu_seqlens_subseq_split=cu_seqlens_split_flat,
        S_split=S_split_total,
        chunk_size=chunk_size,
    )

    initial_states_merge, num_non_first = intracard_merge(
        hm=hm,
        split_info=split_info,
        num_non_first=num_non_first,
        merge_seq_offsets=merge_seq_offsets,
        merge_init_offsets=merge_init_offsets,
        device=device,
        initial_state=initial_state,
        state_v_first=state_v_first,
    )

    if state_v_first:
        initial_state_expanded = k.new_zeros(total_subseqs, HV, V, K, dtype=torch.float32)
    else:
        initial_state_expanded = k.new_zeros(total_subseqs, HV, K, V, dtype=torch.float32)

    # first subsequences inherit h0; later subsequences inherit the preceding affine prefix
    if initial_state is not None:
        initial_state_expanded[first_subseq_indices] = initial_state

    if initial_states_merge is not None and num_non_first > 0:
        initial_state_expanded[non_first_indices] = initial_states_merge

    chunk_indices_subseq = prepare_chunk_indices(cu_seqlens_subseq_gpu, chunk_size)

    h, v_new, final_state_subseq = _raw_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state_expanded,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens_subseq_gpu,
        chunk_indices=chunk_indices_subseq,
        state_v_first=state_v_first,
    )

    # expose one final state per original sequence, taken from its last subsequence
    if output_final_state and final_state_subseq is not None:
        final_state = final_state_subseq[last_subseq_indices]
    else:
        final_state = final_state_subseq

    return h, v_new, final_state
