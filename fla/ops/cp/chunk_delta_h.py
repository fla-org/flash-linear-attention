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

from fla.backends import dispatch
from fla.ops.cp.chunk_bwd import chunk_cp_bwd_dh
from fla.ops.cp.chunk_fwd import chunk_cp_fwd_h
from fla.ops.cp.chunk_merge import chunk_cp_merge
from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.cp.context import _check_cp_device
from fla.utils import IS_NPU

if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext


def _graph_metadata(cu_seqlens: torch.Tensor, context: FLACPContext | None = None, layout: str = 'contiguous'):
    if context is not None:
        layout = context.layout
    if layout == 'zigzag' and not IS_NPU:
        raise NotImplementedError('zigzag CP graph replay is currently supported only on Ascend NPU.')
    metadata = getattr(context, '_cp_graph_metadata', None)
    if metadata is not None:
        return metadata
    if IS_NPU:
        from fla.ops.cp.backends.triton_ascend.graph_metadata import prepare_cp_graph_metadata
        return prepare_cp_graph_metadata(cu_seqlens, layout=layout)
    # contiguous graphs on CUDA also need actual row positions, rather than live-row counts.
    num_seqs = cu_seqlens.numel() - 1
    rows = torch.arange(num_seqs, device=cu_seqlens.device, dtype=torch.int64)
    active = cu_seqlens[1:] > cu_seqlens[:-1]
    first = torch.where(active, rows, num_seqs).amin().reshape(1)
    first = torch.where(first < num_seqs, first, 0)
    last = torch.where(active, rows, -1).amax().reshape(1)
    first_window = cu_seqlens.index_select(0, torch.cat((first, first + 1)))
    last_window = cu_seqlens.index_select(0, torch.cat((last.clamp_min(0), last.clamp_min(0) + 1)))
    return first, last, (first_window,), (last_window,)


@dispatch('cp')
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
            assert context.pre_num_ranks_dev is not None, "use_graph with CP requires context.pre_num_ranks_dev"
            first, _, _, last_windows = _graph_metadata(cu_seqlens, context)
        else:
            last_windows = (cu_seqlens[context.front_num_seqs - 1: context.front_num_seqs + 1], cu_seqlens[-2:])
        front_num_seqs = context.front_num_seqs
        hm = k.new_zeros(2, HV, K, V + K, dtype=torch.float32)
        if state_v_first:
            initial_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            initial_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)
        # each part exports the affine chain of its last segment, skipped when chain-last
        for i_part, (last_window, is_last) in enumerate(zip(
            last_windows,
            (False, False) if use_graph else context.is_last_by_part,
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
                    cu_seqlens=last_window,
                    chunk_size=chunk_size,
                )
        ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
        # chain order: rank fronts in ascending order, then rank backs in descending order
        hm_chain = torch.cat([ag_hm[:, 0], ag_hm[:, 1].flip(0)])
        world_size = dist.get_world_size(group=context.group)

        for i_part in range(2):
            if use_graph or not context.is_first_by_part[i_part]:
                chunk_cp_merge(
                    h=initial_state if use_graph else initial_state[0] if i_part == 0 else initial_state[front_num_seqs],
                    ag_hm=hm_chain,
                    num_ranks=context.pre_num_ranks_dev[i_part:i_part + 1] if use_graph else
                    context.pre_num_ranks_by_part[i_part],
                    rank=rank if i_part == 0 else 2 * world_size - 1 - rank,
                    state_v_first=state_v_first,
                    forward=True,
                    h_seq_idx=first[i_part:i_part + 1] if use_graph else None,
                    use_graph=use_graph,
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
        first, _, _, last_windows = _graph_metadata(cu_seqlens, context)
        cu_seqlens_last = last_windows[0]
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
            h=initial_state if use_graph else initial_state[0],
            ag_hm=ag_hm,
            num_ranks=context.pre_num_ranks_dev if use_graph else context.pre_num_ranks,
            rank=rank,
            state_v_first=state_v_first,
            forward=True,
            h_seq_idx=first if use_graph else None,
            use_graph=use_graph,
        )
    return initial_state


@dispatch('cp')
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
            assert context.post_num_ranks_dev is not None, "use_graph with CP requires context.post_num_ranks_dev"
            _, last, first_windows, _ = _graph_metadata(cu_seqlens, context)
        else:
            first_windows = (cu_seqlens[:2], cu_seqlens[context.front_num_seqs: context.front_num_seqs + 2])
        front_num_seqs = context.front_num_seqs
        dhm = q.new_zeros(2, HV, K, V + K, dtype=torch.float32)
        if state_v_first:
            dht = q.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)
        # each part exports the backward affine chain of its first segment, skipped when chain-first
        for i_part, (first_window, is_first) in enumerate(zip(
            first_windows,
            (False, False) if use_graph else context.is_first_by_part,
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
                    cu_seqlens=first_window,
                    chunk_size=chunk_size,
                )
        ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)
        # chain order: rank fronts in ascending order, then rank backs in descending order
        dhm_chain = torch.cat([ag_dhm[:, 0], ag_dhm[:, 1].flip(0)])
        world_size = dist.get_world_size(group=context.group)

        for i_part in range(2):
            if use_graph or not context.is_last_by_part[i_part]:
                chunk_cp_merge(
                    h=dht if use_graph else dht[front_num_seqs - 1] if i_part == 0 else dht[-1],
                    ag_hm=dhm_chain,
                    num_ranks=context.post_num_ranks_dev[i_part:i_part + 1] if use_graph else
                    context.post_num_ranks_by_part[i_part],
                    rank=rank if i_part == 0 else 2 * world_size - 1 - rank,
                    state_v_first=state_v_first,
                    forward=False,
                    h_seq_idx=last[i_part:i_part + 1] if use_graph else None,
                    use_graph=use_graph,
                )
        return dht, None

    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        dht = q.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)

    cu_seqlens_first = cu_seqlens[:2]
    if use_graph:
        assert context.post_num_ranks_dev is not None, "use_graph with CP requires context.post_num_ranks_dev"
        _, last, first_windows, _ = _graph_metadata(cu_seqlens, context)
        cu_seqlens_first = first_windows[0]
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
            cu_seqlens=cu_seqlens_first,
            chunk_size=chunk_size,
        )

    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)

    # a zero-rank merge would only re-store the existing zeros
    if use_graph or not context.is_last_rank:
        h_seq_idx = None
        h = dht[-1]
        if use_graph:
            h_seq_idx = last
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


def compress_h0(h0: torch.Tensor | None, context: FLACPContext, use_graph: bool = False) -> torch.Tensor | None:
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    if use_graph:
        first, _, _, _ = _graph_metadata(context.cu_seqlens, context)
        return h0.index_select(0, first)
    if context.layout == 'zigzag':
        if len(context.cu_seqlens) == 3:
            # both parts can carry a cross-rank state
            return h0
        # only the first segment of each part can carry a cross-rank state
        return h0[[0, context.front_num_seqs]]
    # clone so autograd does not retain storage for all local sequences
    return h0[:1].clone()


def expand_h0(h0: torch.Tensor | None, context: FLACPContext, use_graph: bool = False) -> torch.Tensor | None:
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    B = len(context.cu_seqlens) - 1
    if use_graph:
        first, _, _, _ = _graph_metadata(context.cu_seqlens, context)
        expanded = h0.new_zeros(B, *h0.shape[1:])
        return expanded.index_copy_(0, first, h0)
    if context.layout == 'zigzag':
        if B == 2:
            return h0
        expanded = h0.new_zeros(B, *h0.shape[1:])
        expanded[[0, context.front_num_seqs]] = h0
        return expanded
    expanded = h0.new_zeros(B, *h0.shape[1:])
    expanded[:1] = h0
    return expanded
