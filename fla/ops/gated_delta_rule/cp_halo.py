# -*- coding: utf-8 -*-
# Utilities for context-parallel halo exchange for short conv pre-processing in GDN

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist


def halo_exchange_and_extend(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    h: int,
    *,
    cp_rank: int,
    cp_size: int,
    cp_group,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cp_shard_start_idx: Optional[int] = None,
    detach_send: bool = True,
    blocking_send: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.LongTensor]]:
    """
    Perform left halo exchange across context-parallel ranks and return extended inputs
    as well as an adjusted cu_seqlens to account for the prepended halo.

    Args:
        q_in, k_in, v_in: [B, T, Dq/Dk/Dv]
        h: halo size (conv_size - 1)
        cp_rank, cp_size, cp_group: CP info
        cu_seqlens: [N+1] cumulative sequence lengths for varlen. If provided, extended
            boundaries will be shifted by +h (except the first 0).
        cp_shard_start_idx: global start index (into flattened tokens) of this CP shard.
            If provided and equals a true sequence boundary, the received halo is zeroed
            to avoid context leakage across sequences.
        detach_send: whether to detach the send buffer from autograd graph.
        blocking_send: when True, uses blocking dist.send; otherwise returns after isend
            is posted (caller must ensure buffer lifetime until completion).

    Returns:
        q_ext, k_ext, v_ext: [B, T+h, ...]
        cu_seqlens_ext: adjusted cu_seqlens (or None)
    """
    assert q_in.dim() == 3 and k_in.dim() == 3 and v_in.dim() == 3
    if h <= 0 or cp_size == 1:
        return q_in, k_in, v_in, cu_seqlens

    B, T = q_in.shape[0], q_in.shape[1]
    Dq = q_in.shape[-1]
    Dk = k_in.shape[-1]
    Dv = v_in.shape[-1]

    # Receive left halo from previous rank (if any)
    if cp_rank > 0:
        halo_recv = q_in.new_empty(B, h, Dq + Dk + Dv)
        recv_req = dist.irecv(halo_recv, src=cp_rank - 1, group=cp_group)
    else:
        halo_recv = None
        recv_req = None

    # Send right tail to next rank (if any)
    if cp_rank < cp_size - 1:
        # Build exactly h tokens: left-pad zeros if T < h
        t_send = min(T, h)
        pad = h - t_send
        tail_q = q_in[:, -t_send:, :]
        tail_k = k_in[:, -t_send:, :]
        tail_v = v_in[:, -t_send:, :]
        if pad > 0:
            zq = q_in.new_zeros(B, pad, Dq)
            zk = k_in.new_zeros(B, pad, Dk)
            zv = v_in.new_zeros(B, pad, Dv)
            tail_q = torch.cat([zq, tail_q], dim=1)
            tail_k = torch.cat([zk, tail_k], dim=1)
            tail_v = torch.cat([zv, tail_v], dim=1)
        send_buf = torch.cat([tail_q, tail_k, tail_v], dim=-1).contiguous()  # [B, h, Dq+Dk+Dv]
        if detach_send:
            send_buf = send_buf.detach()
        if blocking_send:
            dist.send(send_buf, dst=cp_rank + 1, group=cp_group)
        else:
            # Caller must keep send_buf alive until work.wait()
            work = dist.isend(send_buf, dst=cp_rank + 1, group=cp_group)
            work.wait()

    # Prepare halos
    if cp_rank > 0:
        recv_req.wait()
        q_halo, k_halo, v_halo = torch.split(halo_recv, [Dq, Dk, Dv], dim=-1)
        # If this shard begins at a true sequence boundary, zero halo to avoid leakage
        if cu_seqlens is not None and cp_shard_start_idx is not None:
            seq_boundaries = cu_seqlens[1:-1]
            if seq_boundaries.numel() > 0 and bool((seq_boundaries == cp_shard_start_idx).any().item()):
                q_halo.zero_(); k_halo.zero_(); v_halo.zero_()
    else:
        q_halo = q_in.new_zeros(B, h, Dq)
        k_halo = k_in.new_zeros(B, h, Dk)
        v_halo = v_in.new_zeros(B, h, Dv)

    # Concatenate halo with local inputs
    q_ext = torch.cat([q_halo, q_in], dim=1)
    k_ext = torch.cat([k_halo, k_in], dim=1)
    v_ext = torch.cat([v_halo, v_in], dim=1)

    # Adjust cu_seqlens to account for the prepended halo
    cu_seqlens_ext = None
    if cu_seqlens is not None:
        cu_seqlens_ext = cu_seqlens.clone()
        if h > 0:
            cu_seqlens_ext[1:] = cu_seqlens_ext[1:] + h

    return q_ext, k_ext, v_ext, cu_seqlens_ext
