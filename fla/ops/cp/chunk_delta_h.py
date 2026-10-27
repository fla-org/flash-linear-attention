# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from fla.ops.cp.chunk_bwd import chunk_cp_bwd_dh
from fla.ops.cp.chunk_fwd import chunk_cp_fwd_h
from fla.ops.cp.chunk_merge import chunk_cp_merge
from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.cp.context import _check_cp_device

if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext


def chunk_gated_delta_rule_fwd_h_pre_process(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    chunk_size: int = 64,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
    use_graph: bool = False,
) -> torch.Tensor | None:
    if context is None or context.group is None:
        return initial_state
    assert initial_state is None, "When enable CP, the provided initial_state must be None."
    _check_cp_device(k.device)
    rank = dist.get_rank(group=context.group)

    B, _, _, K = k.shape
    HV, V = u.shape[2:]

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if context.layout == 'zigzag':
        if use_graph:
            raise NotImplementedError("use_graph is not supported with zigzag CP layout")
        front_num_seqs = context.front_num_seqs
        hm = k.new_zeros(2, HV, K, V + K, dtype=torch.float32)
        if state_v_first:
            initial_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            initial_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)
        # each part exports the affine chain of its last segment, skipped when chain-last
        for i_part, (cu_seqlens_part, is_last) in enumerate(zip(
            (cu_seqlens[front_num_seqs - 1: front_num_seqs + 1], cu_seqlens[-2:]),
            context.is_last_by_part,
        )):
            if not is_last:
                chunk_cp_fwd_h(
                    k=k,
                    w=w,
                    u=u,
                    hm=hm[i_part],
                    g=g,
                    gk=gk,
                    bg=bg,
                    v=v,
                    cu_seqlens=cu_seqlens_part,
                    chunk_size=chunk_size,
                )
        ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
        # chain order: rank fronts in ascending order, then rank backs in descending order
        hm_chain = torch.cat([ag_hm[:, 0], ag_hm[:, 1].flip(0)])
        world_size = dist.get_world_size(group=context.group)

        for i_part in range(2):
            if not context.is_first_by_part[i_part]:
                chunk_cp_merge(
                    h=initial_state[0] if i_part == 0 else initial_state[front_num_seqs],
                    ag_hm=hm_chain,
                    num_ranks=context.pre_num_ranks_by_part[i_part],
                    rank=rank if i_part == 0 else 2 * world_size - 1 - rank,
                    state_v_first=state_v_first,
                    forward=True,
                    h_seq_idx=None,
                    use_graph=False,
                )
        return initial_state

    hm = k.new_zeros(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        initial_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        initial_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)
    cu_seqlens_last = cu_seqlens[-2:]
    if use_graph:
        assert context.pre_num_ranks_dev is not None, "use_graph with CP requires context.pre_num_ranks_dev"
        # skip zero-length tail padding when selecting the last sequence during graph replay
        i_last = (cu_seqlens[1:] > cu_seqlens[:-1]).sum() - 1
        cu_seqlens_last = cu_seqlens.index_select(0, torch.stack((i_last, i_last + 1)))
    # every rank participates during graph replay because host-side flags are frozen at capture
    if use_graph or not context.is_last_rank:
        chunk_cp_fwd_h(
            k=k,
            w=w,
            u=u,
            hm=hm,
            g=g,
            gk=gk,
            bg=bg,
            v=v,
            cu_seqlens=cu_seqlens_last,
            chunk_size=chunk_size,
        )
    ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
    # a zero-rank merge would only re-store the existing zeros
    if use_graph or not context.is_first_rank:
        chunk_cp_merge(
            h=initial_state[0],
            ag_hm=ag_hm,
            num_ranks=context.pre_num_ranks_dev if use_graph else context.pre_num_ranks,
            rank=rank,
            state_v_first=state_v_first,
            forward=True,
            h_seq_idx=None,
            use_graph=use_graph,
        )
    return initial_state


def chunk_gated_delta_rule_bwd_dhu_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
    chunk_size: int = 64,
    use_graph: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if context is None or context.group is None:
        return dht, initial_state
    assert dht is None, "When enable CP, the provided dht must be None."
    _check_cp_device(k.device)
    rank = dist.get_rank(group=context.group)

    B, _, _, K = q.shape
    HV, V = do.shape[2:]
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    if context.layout == 'zigzag':
        if use_graph:
            raise NotImplementedError("use_graph is not supported with zigzag CP layout")
        front_num_seqs = context.front_num_seqs
        dhm = q.new_zeros(2, HV, K, V + K, dtype=torch.float32)
        if state_v_first:
            dht = q.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)
        # each part exports the backward affine chain of its first segment, skipped when chain-first
        for i_part, (cu_seqlens_part, is_first) in enumerate(zip(
            (cu_seqlens[:2], cu_seqlens[front_num_seqs: front_num_seqs + 2]),
            context.is_first_by_part,
        )):
            if not is_first:
                chunk_cp_bwd_dh(
                    q=q,
                    k=k,
                    w=w,
                    do=do,
                    dv=dv,
                    dhm=dhm[i_part],
                    g=g,
                    gk=gk,
                    bg=bg,
                    scale=scale,
                    cu_seqlens=cu_seqlens_part,
                    chunk_size=chunk_size,
                )
        ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)
        # chain order: rank fronts in ascending order, then rank backs in descending order
        dhm_chain = torch.cat([ag_dhm[:, 0], ag_dhm[:, 1].flip(0)])
        world_size = dist.get_world_size(group=context.group)

        for i_part in range(2):
            if not context.is_last_by_part[i_part]:
                chunk_cp_merge(
                    h=dht[front_num_seqs - 1] if i_part == 0 else dht[-1],
                    ag_hm=dhm_chain,
                    num_ranks=context.post_num_ranks_by_part[i_part],
                    rank=rank if i_part == 0 else 2 * world_size - 1 - rank,
                    state_v_first=state_v_first,
                    forward=False,
                    h_seq_idx=None,
                    use_graph=False,
                )
        return dht, None

    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        dht = q.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)

    # every rank participates during graph replay because host-side flags are frozen at capture
    if use_graph or not context.is_first_rank:
        chunk_cp_bwd_dh(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            dhm=dhm,
            g=g,
            gk=gk,
            bg=bg,
            scale=scale,
            cu_seqlens=cu_seqlens[:2],
            chunk_size=chunk_size,
        )

    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)

    # a zero-rank merge would only re-store the existing zeros
    if use_graph or not context.is_last_rank:
        h_seq_idx = None
        h = dht[-1]
        if use_graph:
            assert context.post_num_ranks_dev is not None, "use_graph with CP requires context.post_num_ranks_dev"
            # skip zero-length tail padding when selecting the merge target during graph replay
            h_seq_idx = (cu_seqlens[1:] > cu_seqlens[:-1]).sum() - 1
            h = dht

        chunk_cp_merge(
            h=h,
            ag_hm=ag_dhm,
            num_ranks=context.post_num_ranks_dev if use_graph else context.post_num_ranks,
            rank=rank,
            state_v_first=state_v_first,
            forward=False,
            h_seq_idx=h_seq_idx,
            use_graph=use_graph,
        )

    return dht, None


def compress_h0(h0: torch.Tensor | None, context: FLACPContext) -> torch.Tensor | None:
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    if context.layout == 'zigzag':
        if len(context.cu_seqlens) == 3:
            # both parts can carry a cross-rank state
            return h0
        # only the first segment of each part can carry a cross-rank state
        return h0[[0, context.front_num_seqs]]
    # clone so autograd does not retain storage for all local sequences
    return h0[:1].clone()


def expand_h0(h0: torch.Tensor | None, context: FLACPContext) -> torch.Tensor | None:
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    B = len(context.cu_seqlens) - 1
    if context.layout == 'zigzag':
        if B == 2:
            return h0
        expanded = h0.new_zeros(B, *h0.shape[1:])
        expanded[[0, context.front_num_seqs]] = h0
        return expanded
    expanded = h0.new_zeros(B, *h0.shape[1:])
    expanded[:1] = h0
    return expanded
