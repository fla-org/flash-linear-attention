"""Intra-Card Context Parallel for KDA inference (varlen mode only).

Optimized: all CPU-side index computation uses pure Python loops instead of
torch tensor operations (repeat_interleave, arange, cumsum, etc.) to eliminate
per-op overhead on tiny arrays. GPU tensors are created directly from Python
lists to minimize cudaStreamSynchronize calls.
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
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import get_multiprocessor_count

logger = logging.getLogger(__name__)


# Cache for intracard_fwd_h precomputation (Python results + GPU tensors)
# Key: object id of cu_seqlens (consistent with tensor_cache philosophy)
_intracard_cache: OrderedDict[tuple, _CacheEntry] = OrderedDict()
_INTRACARD_CACHE_MAXSIZE = 32


class _CacheEntry(NamedTuple):
    """Cache entry for intracard_fwd_h precomputation.

    Caches both Python computation results and GPU tensors to eliminate
    redundant CPU→GPU transfers and Python loop computation.
    """
    # Keep a weak reference to validate id-based key safety.
    # If Python reuses an object id after GC, this guard prevents stale hits.
    cu_seqlens_ref: weakref.ReferenceType[torch.Tensor]
    # From prepare_subseq_cu_seqlens
    cu_seqlens_subseq_values: list[int]
    split_info: SplitSeqInfo
    total_subseqs: int
    # From _precompute_intracard_indices
    cu_seqlens_split_values: list[int]
    S_split_total: int
    non_first_indices: list[int]
    first_subseq_indices: list[int]
    last_subseq_indices: list[int]
    num_non_first: int
    merge_seq_offsets: list[int]
    merge_init_offsets: list[int]
    # GPU tensors (cached to avoid H2D transfer)
    cu_seqlens_subseq_gpu: torch.Tensor
    cu_seqlens_split_flat: torch.Tensor


class SplitSeqInfo(NamedTuple):
    """Information about split sequences (Python lists for zero-overhead access)."""
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
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_zeros(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k, v=u, w=w, v_new=v_new,
        g=g, gk=gk, h=h, h0=initial_state, ht=final_state,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        T=T, H=H, K=K, V=V, BT=BT, USE_EXP2=use_exp2,
    )
    return h, v_new, final_state


def compute_subseq_len(
    seq_len: int,
    num_sms: int,
    num_heads: int,
    chunk_size: int = 64,
) -> int:
    """Compute sub-sequence length for intracard splitting.

    For linear recurrence (fwd_h), the sequential scan is the bottleneck.
    Splitting always reduces the critical path and helps, as long as the
    sequence is long enough to amortize the pre_scan + merge overhead.

    The fwd_h kernel grid is (num_v_blocks, N*H) where num_v_blocks ≈ 2.
    Each sub-sequence contributes 2*H blocks. We target enough splits so
    that even a single long sequence can saturate all SMs.

    A floor on subseq_chunks (MIN_SUBSEQ_CHUNKS) prevents subseq_len from
    being too small, which would cause prepare_subseq_cu_seqlens to
    unnecessarily split shorter sequences in mixed-length batches
    (split threshold = 2 * subseq_len).
    """
    seq_chunks = (seq_len + chunk_size - 1) // chunk_size

    if seq_chunks < 8:
        return seq_len

    # Target splits: saturate SMs with the longest sequence alone.
    # Each sub-seq contributes NUM_V_BLOCKS * num_heads blocks.
    # Always at least 4 — for linear recurrence, CP4 always helps.
    NUM_V_BLOCKS = 2
    target_splits = max(4, num_sms // (NUM_V_BLOCKS * num_heads))

    subseq_chunks = (seq_chunks + target_splits - 1) // target_splits

    # Floor: prevent subseq_len from being too small.
    # With chunk_size=64, MIN_SUBSEQ_CHUNKS=128 → subseq_len >= 8192 tokens,
    # split threshold (3 * subseq_len) = 24576 tokens.
    # Sequences shorter than it won't be split.
    MIN_SUBSEQ_CHUNKS = 128
    subseq_chunks = max(subseq_chunks, MIN_SUBSEQ_CHUNKS)

    return subseq_chunks * chunk_size


def prepare_subseq_cu_seqlens(
    cu_seqlens_cpu: torch.Tensor,
    subseq_len: int,
    chunk_size: int = 64,
    max_splits: int = 32,
) -> tuple[list[int], SplitSeqInfo | bool, int]:
    """Insert subseq split points into original cu_seqlens.

    Optimized: uses pure Python loops instead of torch tensor operations
    for the small index arrays (typically 1-32 elements).

    Returns:
        boundaries: List of cu_seqlens boundaries (can be used directly by _precompute_intracard_indices)
        split_info: SplitSeqInfo for sequences that need splitting, or False if no splitting needed
        total_subseqs: Total number of sub-sequences after splitting
    """
    N = len(cu_seqlens_cpu) - 1
    if N == 0:
        return cu_seqlens_cpu.tolist(), False, 0

    subseq_chunks = (subseq_len + chunk_size - 1) // chunk_size
    threshold_subseq_len = 3 * subseq_len

    split_seq_ids: list[int] = []
    start_subseq_idxs: list[int] = []
    num_subseqs_list: list[int] = []

    # Build boundaries using pure Python loop
    boundaries: list[int] = [0]
    cumsum_offset = 0

    for i in range(N):
        seq_start = int(cu_seqlens_cpu[i].item())
        seq_end = int(cu_seqlens_cpu[i + 1].item())
        seq_len_i = seq_end - seq_start
        seq_chunks_i = (seq_len_i + chunk_size - 1) // chunk_size

        if seq_len_i >= threshold_subseq_len:
            # This sequence needs splitting
            num_ss = min(max_splits, (seq_chunks_i + subseq_chunks - 1) // subseq_chunks)
            chunks_per = (seq_chunks_i + num_ss - 1) // num_ss
            actual_ssl = chunks_per * chunk_size

            split_seq_ids.append(i)
            start_subseq_idxs.append(cumsum_offset)
            num_subseqs_list.append(num_ss)

            for j in range(num_ss):
                boundary = min(seq_start + (j + 1) * actual_ssl, seq_end)
                boundaries.append(boundary)
            cumsum_offset += num_ss
        else:
            # No split needed, single sub-sequence
            boundaries.append(seq_end)
            cumsum_offset += 1

    if not split_seq_ids:
        return cu_seqlens_cpu.tolist(), False, 0

    total_subseqs = cumsum_offset

    split_info = SplitSeqInfo(
        split_seq_ids=split_seq_ids,
        start_subseq_idx=start_subseq_idxs,
        num_subseqs=num_subseqs_list,
    )

    return boundaries, split_info, total_subseqs


def intracard_pre_scan(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    cu_seqlens_subseq_split: torch.Tensor,
    S_split: int,
    chunk_size: int = 64,
    use_exp2: bool = True,
):
    H, K, V = kg.shape[2], kg.shape[3], u.shape[3]
    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64

    hm = kg.new_empty(S_split, H, K, V + K, dtype=torch.float32)

    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H, S_split)
    pre_process_fwd_kernel_merged[grid](
        k=kg,
        v=u,
        w=w,
        g=None,
        gk=gk,
        hm=hm,
        cu_seqlens=cu_seqlens_subseq_split,
        T=0,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BK1=BK,
        USE_EXP2=use_exp2,
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
) -> tuple[torch.Tensor | None, int]:
    """Merge sub-sequence states using pre-computed parameters.

    All CPU-side preparation (cumsum, offset lists) is done in the caller
    using pure Python loops. This function only creates GPU tensors and
    launches the merge kernel.
    """
    from fla.ops.cp.chunk_delta_h import merge_fwd_bwd_kernel

    if num_non_first == 0:
        return None, 0

    H = hm.shape[1]
    K = hm.shape[2]
    V = hm.shape[3] - K
    BK = triton.next_power_of_2(K)

    num_split_seqs = split_info.num_split_seqs

    # Create all small GPU tensors from Python lists in one batch
    # Merge into a single CPU→GPU transfer to minimize cudaStreamSynchronize
    all_int_data = merge_seq_offsets + merge_init_offsets + split_info.split_seq_ids
    all_tensor = torch.tensor(all_int_data, dtype=torch.int32, device=device)
    n_so = len(merge_seq_offsets)
    n_io = len(merge_init_offsets)
    seq_offsets = all_tensor[:n_so]
    init_offsets = all_tensor[n_so:n_so + n_io]
    h0_seq_ids = all_tensor[n_so + n_io:]

    initial_states_merge = hm.new_empty(num_non_first, H, K, V, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), num_split_seqs, H)

    merge_fwd_bwd_kernel[grid](
        h=initial_states_merge,
        ag_hm=hm,
        pre_or_post_num_ranks=num_split_seqs,
        rank=0,
        seq_offsets=seq_offsets,
        init_offsets=init_offsets,
        h0_seq_ids=h0_seq_ids,
        h0=initial_state,
        H=H,
        K=K,
        V=V,
        BK=BK,
        FORWARD=True,
        INTRACARD_MODE=True,
        NUM_SEQ_ENTRIES=num_split_seqs,
    )

    return initial_states_merge, num_non_first


def _precompute_intracard_indices(
    split_info: SplitSeqInfo,
    cu_seqlens_subseq_values: list[int],
    N_orig: int,
) -> tuple[list[int], int, list[int], list[int], list[int], int, list[int], list[int]]:
    """Pre-compute all derived indices using pure Python loops.

    Returns:
        cu_seqlens_split_values: flattened cu_seqlens boundaries for split seqs (for pre_scan)
        S_split_total: total number of sub-sequences from splits
        non_first_indices: indices for scattering merge results into initial_state_expanded
        first_subseq_indices: indices of first sub-sequence for each original sequence
        last_subseq_indices: indices of last sub-sequence for each original sequence
        num_non_first: total non-first sub-sequences (merge work)
        merge_seq_offsets: cumulative sub-sequence counts for merge kernel
        merge_init_offsets: cumulative non-first counts for merge kernel
    """
    starts = split_info.start_subseq_idx
    num_ss = split_info.num_subseqs
    split_ids = split_info.split_seq_ids

    # cu_seqlens_split_values: for each split seq, extract [start:start+n+1] boundaries
    cu_seqlens_split_values: list[int] = []
    S_split_total = 0
    for s, n in zip(starts, num_ss):
        cu_seqlens_split_values.extend(cu_seqlens_subseq_values[s:s + n + 1])
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

    # merge parameters
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
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    max_splits: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert cu_seqlens is not None, "intracard_fwd_h requires cu_seqlens"

    _, _, H, K, V = *k.shape, u.shape[-1]
    device = k.device

    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()

    seq_lens = torch.diff(cu_seqlens_cpu)
    max_seq_len = int(seq_lens.max().item())
    num_sms = get_multiprocessor_count()
    subseq_len = compute_subseq_len(max_seq_len, num_sms, H, chunk_size)

    early_return = (seq_lens < 2 * subseq_len).all()

    cached = None
    cache_key = None

    if not early_return:
        # Use object identity (id) for cache key, consistent with tensor_cache philosophy
        # vLLM slice creates new Python objects per batch, so id(cu_seqlens) is safe
        cache_key = (
            id(cu_seqlens),  # Object identity, not content hash
            subseq_len,
            chunk_size,
            max_splits,
            str(device),
        )
        cached = _intracard_cache.get(cache_key)
        if cached is not None:
            # Guard against rare Python id reuse after original tensor is GC-ed.
            # We only consider it a hit when the weakref points to the current object.
            if cached.cu_seqlens_ref() is cu_seqlens:
                _intracard_cache.move_to_end(cache_key)
            else:
                _intracard_cache.pop(cache_key, None)
                cached = None

        if cached is not None:
            # Cache hit: reuse all precomputed results including GPU tensors
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
            # Cache miss: compute Python lists
            cu_seqlens_subseq_values, split_info, total_subseqs = prepare_subseq_cu_seqlens(
                cu_seqlens_cpu, subseq_len, chunk_size, max_splits=max_splits
            )

    if early_return or not split_info:
        return _raw_chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g, gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
        )

    N_orig = len(cu_seqlens_cpu) - 1

    if cached is None:
        # Cache miss: continue Python computation and create GPU tensors
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

        # Create GPU tensors (will be cached for reuse)
        dtype = cu_seqlens_cpu.dtype
        cu_seqlens_subseq_gpu = torch.tensor(cu_seqlens_subseq_values, dtype=dtype, device=device)
        cu_seqlens_split_flat = torch.tensor(cu_seqlens_split_values, dtype=dtype, device=device)

        # Store all results in cache (including GPU tensors to avoid H2D)
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
        # Evict oldest entries if over capacity
        while len(_intracard_cache) > _INTRACARD_CACHE_MAXSIZE:
            _intracard_cache.popitem(last=False)

    hm = intracard_pre_scan(
        kg=k, w=w, u=u, gk=gk,
        cu_seqlens_subseq_split=cu_seqlens_split_flat,
        S_split=S_split_total,
        chunk_size=chunk_size,
        use_exp2=use_exp2,
    )

    initial_states_merge, num_non_first = intracard_merge(
        hm=hm,
        split_info=split_info,
        num_non_first=num_non_first,
        merge_seq_offsets=merge_seq_offsets,
        merge_init_offsets=merge_init_offsets,
        device=device,
        initial_state=initial_state,
    )

    initial_state_expanded = k.new_zeros(total_subseqs, H, K, V, dtype=torch.float32)

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
        use_exp2=use_exp2,
    )

    if output_final_state and final_state_subseq is not None:
        final_state = final_state_subseq[last_subseq_indices]
    else:
        final_state = final_state_subseq

    return h, v_new, final_state
