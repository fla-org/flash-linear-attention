"""Intra-Card Context Parallel for KDA inference (varlen mode only)."""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import triton

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from fla.ops.cp.chunk_delta_h import pre_process_fwd_kernel_merged
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets

logger = logging.getLogger(__name__)


class SplitSeqInfo(NamedTuple):
    """Information about split sequences (CPU tensors)."""
    split_seq_ids: torch.Tensor  # [num_split_seqs]
    start_subseq_idx: torch.Tensor  # [num_split_seqs]
    num_subseqs: torch.Tensor  # [num_split_seqs]

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
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
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
    """Compute subseq_len based on SM overflow per head.

    Logic: If chunks_per_head > num_sms, we need to split to utilize all SMs.
    Each subseq should have enough chunks to saturate all SMs.
    """
    seq_chunks = (seq_len + chunk_size - 1) // chunk_size
    chunks_per_head = seq_chunks / num_heads  # Use float for comparison

    if chunks_per_head <= num_sms:
        # No overflow, no need to split
        return seq_len  # Return full seq len = no split

    # Overflow: compute subseq_len to fill all SMs
    # Target: each subseq has num_sms chunks to saturate all SMs
    target_subseq_chunks = num_sms
    subseq_len = target_subseq_chunks * chunk_size
    return subseq_len


def prepare_subseq_cu_seqlens(
    cu_seqlens_cpu: torch.Tensor,
    subseq_len: int,
    chunk_size: int = 64,
    max_splits: int = 32,
) -> tuple[torch.Tensor, SplitSeqInfo, int]:
    """Insert subseq split points into original cu_seqlens."""
    N = len(cu_seqlens_cpu) - 1
    if N == 0:
        return cu_seqlens_cpu.clone(), SplitSeqInfo(
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32)
        ), 0

    seq_starts = cu_seqlens_cpu[:-1]
    seq_ends = cu_seqlens_cpu[1:]
    seq_lens = seq_ends - seq_starts

    needs_split = seq_lens >= 2 * subseq_len

    seq_chunks = (seq_lens + chunk_size - 1) // chunk_size
    subseq_chunks = (subseq_len + chunk_size - 1) // chunk_size
    num_ss = (seq_chunks + subseq_chunks - 1) // subseq_chunks
    num_ss = torch.clamp(num_ss, max=max_splits)
    num_ss = torch.where(needs_split, num_ss, torch.ones_like(num_ss))

    chunks_per_split = torch.where(
        needs_split,
        (seq_chunks + num_ss - 1) // num_ss,
        seq_chunks
    )
    actual_subseq_len = chunks_per_split * chunk_size

    cumsum_offset = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        num_ss.cumsum(dim=0)[:-1]
    ])

    split_indices = torch.where(needs_split)[0]
    split_info = SplitSeqInfo(
        split_seq_ids=split_indices.cpu(),
        start_subseq_idx=cumsum_offset[split_indices].cpu(),
        num_subseqs=num_ss[split_indices].cpu(),
    )

    total_subseqs = int(num_ss.sum().item())

    seq_ids = torch.repeat_interleave(torch.arange(N, dtype=torch.int32), num_ss)
    local_idx = torch.arange(total_subseqs, dtype=torch.int32) - torch.repeat_interleave(cumsum_offset, num_ss)

    seq_starts_expanded = seq_starts[seq_ids]
    actual_subseq_len_expanded = actual_subseq_len[seq_ids]
    seq_ends_expanded = seq_ends[seq_ids]

    boundaries = seq_starts_expanded + (local_idx + 1) * actual_subseq_len_expanded
    boundaries = torch.min(boundaries, seq_ends_expanded)

    cu_seqlens_subseq = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        boundaries
    ])

    return cu_seqlens_subseq, split_info, total_subseqs


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
    cu_seqlens_subseq: torch.Tensor,
    total_subseqs: int,
    device: torch.device,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    from fla.ops.cp.chunk_delta_h import merge_fwd_bwd_kernel

    H = hm.shape[1]
    K = hm.shape[2]
    V = hm.shape[3] - K
    BK = triton.next_power_of_2(K)

    num_split_seqs = split_info.num_split_seqs
    num_ss_list = split_info.num_subseqs

    seq_offsets_list = [0] + num_ss_list.cumsum(dim=0).tolist()
    non_first_counts = num_ss_list - 1
    init_offsets_list = [0] + non_first_counts.cumsum(dim=0).tolist()

    num_non_first = int(init_offsets_list[-1])
    if num_non_first == 0:
        return None, 0

    h0_seq_ids_list = split_info.split_seq_ids.tolist()

    seq_offsets = torch.tensor(seq_offsets_list, dtype=torch.int32, device=device)
    init_offsets = torch.tensor(init_offsets_list, dtype=torch.int32, device=device)
    h0_seq_ids = torch.tensor(h0_seq_ids_list, dtype=torch.int32, device=device)
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
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    subseq_len = compute_subseq_len(max_seq_len, num_sms, H, chunk_size)

    cu_seqlens_subseq, split_info, total_subseqs = prepare_subseq_cu_seqlens(
        cu_seqlens_cpu, subseq_len, chunk_size, max_splits=max_splits
    )

    N_orig = len(cu_seqlens_cpu) - 1

    if not split_info:
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

    cu_seqlens_subseq_gpu = cu_seqlens_subseq.to(device, non_blocking=True)

    starts = split_info.start_subseq_idx
    num_ss_list = split_info.num_subseqs
    S_split_total = int(num_ss_list.sum().item())

    counts = num_ss_list + 1
    base_indices = torch.repeat_interleave(starts, counts)
    local_offsets = torch.arange(counts.sum(), dtype=torch.int32) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    all_indices = base_indices + local_offsets

    cu_seqlens_split_flat = cu_seqlens_subseq[all_indices].to(device)

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
        cu_seqlens_subseq=cu_seqlens_subseq,
        total_subseqs=total_subseqs,
        device=device,
        initial_state=initial_state,
    )

    # Precompute num_subseqs_per_seq (used for both initial_state scatter and final_state gather)
    num_subseqs_per_seq = torch.ones(N_orig, dtype=torch.int32)
    num_subseqs_per_seq.scatter_(0, split_info.split_seq_ids, split_info.num_subseqs)

    initial_state_expanded = k.new_zeros(total_subseqs, H, K, V, dtype=torch.float32)

    if initial_state is not None:
        first_subseq_indices = torch.cat([
            torch.zeros(1, dtype=torch.int32),
            num_subseqs_per_seq[:-1].cumsum(dim=0)
        ])
        initial_state_expanded[first_subseq_indices] = initial_state

    if initial_states_merge is not None and num_non_first > 0:
        split_starts = split_info.start_subseq_idx
        num_ss_tensor = split_info.num_subseqs
        non_first_counts = num_ss_tensor - 1
        base_indices = torch.repeat_interleave(split_starts, non_first_counts)
        non_first_offsets = torch.arange(non_first_counts.sum(), dtype=torch.int32) - torch.repeat_interleave(
            non_first_counts.cumsum(0) - non_first_counts, non_first_counts
        ) + 1
        non_first_indices = base_indices + non_first_offsets
        initial_state_expanded[non_first_indices] = initial_states_merge

    chunk_indices_subseq = prepare_chunk_indices(cu_seqlens_subseq_gpu, chunk_size, cu_seqlens_cpu=cu_seqlens_subseq)

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
        last_subseq_indices = num_subseqs_per_seq.cumsum(dim=0) - 1
        final_state = final_state_subseq[last_subseq_indices]
    else:
        final_state = final_state_subseq

    return h, v_new, final_state
