#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for CP halo exchange with autograd.

Covers:
- Forward parity vs. single-rank baseline for fixed-length and varlen.
- T < h and T >= h.
- Backward parity for a simple causal depthwise conv over features.

Run examples:
  # Forward unit parity (single then cp)
  python tests/cp/test_cp_halo_unit.py --mode single --save_ref
  torchrun --nproc_per_node=2 tests/cp/test_cp_halo_unit.py --mode cp --cp_size 2

  # Varlen
  python tests/cp/test_cp_halo_unit.py --mode single --save_ref --varlen
  torchrun --nproc_per_node=2 tests/cp/test_cp_halo_unit.py --mode cp --cp_size 2 --varlen

  # Backward parity
  python tests/cp/test_cp_halo_unit.py --mode single --save_ref --backward
  torchrun --nproc_per_node=2 tests/cp/test_cp_halo_unit.py --mode cp --cp_size 2 --backward
"""

import argparse
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.ops.gated_delta_rule.cp_halo import halo_exchange_and_extend_autograd

REF_PATH = 'tests/cp/.cp_halo_unit_ref.pt'

def seed_all(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_dist(cp_size: int):
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(cp_size)))
    return rank, device, group


def build_depthwise_causal_weight(D: int, k: int, device) -> torch.Tensor:
    """Builds a depthwise conv1d kernel [D,1,k] with deterministic values.
    Causal: weight only uses past including t (no future).
    """
    # Simple ramp kernel shared across channels but repeated depthwise
    base = torch.linspace(1.0, 0.5, steps=k, device=device)
    w = base.repeat(D, 1).unsqueeze(1).contiguous()  # [D,1,k]
    return w


def depthwise_causal_conv(x: torch.Tensor, w: torch.Tensor, h: int) -> torch.Tensor:
    """Apply depthwise conv1d over features.
    x: [B, T, D], w: [D,1,k], k=h+1, returns [B, T, D]
    Assumes left-halo already handled by caller (no extra padding here).
    """
    B, T, D = x.shape
    k = w.shape[-1]
    assert k == h + 1
    # Conv1d expects [B,C,L], groups=D
    y = F.conv1d(x.movedim(1, 2), w, bias=None, stride=1, padding=0, groups=D)  # [B, D, T - k + 1]
    # Because no padding, output length is T-k+1; callers should have extended x with h halo so
    # that after dropping the first h we realign to original length.
    return y.movedim(1, 2)


def global_baseline(x: torch.Tensor, w: torch.Tensor, h: int, cu_seqlens: Optional[torch.LongTensor] = None) -> torch.Tensor:
    """Single-rank baseline. Applies causal conv per sequence segment if varlen, else over full time.
    Returns y of shape [B, T, D].
    """
    B, T, D = x.shape
    k = w.shape[-1]
    assert k == h + 1
    if cu_seqlens is None:
        # Pad left with h zeros to simulate causal context
        x_pad = torch.cat([torch.zeros(B, h, D, device=x.device, dtype=x.dtype), x], dim=1)
        y_ext = depthwise_causal_conv(x_pad, w, h)  # [B, T, D]
        return y_ext
    else:
        # Process each segment independently
        y_parts = []
        for i in range(len(cu_seqlens) - 1):
            s = int(cu_seqlens[i].item())
            e = int(cu_seqlens[i + 1].item())
            xi = x[:, s:e, :]
            xi_pad = torch.cat([torch.zeros(B, h, D, device=x.device, dtype=x.dtype), xi], dim=1)
            yi = depthwise_causal_conv(xi_pad, w, h)
            y_parts.append(yi)
        return torch.cat(y_parts, dim=1)


def run_single(save_ref: bool, varlen: bool, backward: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(123)
    B, T, D = 2, 32, 8
    h_values = [1, 3]  # covers T<h and T>=h when we vary T later if needed
    results = {}

    # Optionally use varlen with boundary at mid
    cu_seqlens = None
    if varlen:
        cu_seqlens = torch.tensor([0, T // 2, T], dtype=torch.long, device=device)

    x = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=backward)

    for h in h_values:
        k = h + 1
        w = build_depthwise_causal_weight(D, k, device)
        w.requires_grad = backward
        y = global_baseline(x, w, h, cu_seqlens)
        loss = y.sum() if backward else None
        gx = None
        gw = None
        if backward:
            loss.backward()
            gx = x.grad.detach().cpu()
            gw = w.grad.detach().cpu()
            x.grad = None
            w.grad = None
        results[h] = {
            'y': y.detach().cpu(),
            'gx': gx,
            'gw': gw,
        }

    ref = {
        'x': x.detach().cpu(),
        'cu_seqlens': cu_seqlens.detach().cpu() if cu_seqlens is not None else None,
        'results': results,
    }
    if save_ref:
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        torch.save(ref, REF_PATH)
        print(f"Saved reference to {REF_PATH}")
    return ref


def run_cp(cp_size: int, varlen: bool, backward: bool):
    rank, device, group = setup_dist(cp_size)

    # Load reference from rank 0
    if rank == 0:
        assert os.path.exists(REF_PATH), f"Run single first to create {REF_PATH}"
        ref = torch.load(REF_PATH, map_location='cpu')
    else:
        ref = None
    obj = [ref]
    dist.broadcast_object_list(obj, src=0, group=group)
    ref = obj[0]

    x_full = ref['x'].to(device)
    cu_full = ref['cu_seqlens']
    cu_full = cu_full.to(device) if cu_full is not None else None

    B, T, D = x_full.shape
    assert T % cp_size == 0
    chunk = T // cp_size
    s = rank * chunk
    e = s + chunk

    x_local = x_full[:, s:e, :].clone().to(device).requires_grad_(backward)

    for h, pack in ref['results'].items():
        k = h + 1
        w = build_depthwise_causal_weight(D, k, device)
        w.requires_grad = backward

        # Prepare q/k/v inputs; reuse x for q, zeros for k/v to keep loss simple
        q_in = x_local
        k_in = torch.zeros_like(q_in)
        v_in = torch.zeros_like(q_in)

        # Autograd-aware halo exchange and simple causal conv
        q_ext, k_ext, v_ext, cu_ext = halo_exchange_and_extend_autograd(
            q_in, k_in, v_in, h,
            cp_rank=rank, cp_size=cp_size, cp_group=group,
            cu_seqlens=cu_full, cp_shard_start_idx=s if cu_full is not None else None,
        )
        # Apply conv on extended tensors; we only use q branch for loss/compare
        y_ext = depthwise_causal_conv(q_ext, w, h)
        y_local = y_ext[:, h:, :]  # drop halo

        # Gather outputs for forward parity
        y_list = [torch.zeros_like(y_local) for _ in range(cp_size)]
        dist.all_gather(y_list, y_local, group=group)
        y_full = torch.cat(y_list, dim=1)

        # Compare forward
        y_ref = pack['y'].to(device)
        diff = (y_full - y_ref).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        if rank == 0:
            print(f"h={h} forward parity: max {max_diff:.6f}, mean {mean_diff:.6f}")
        assert max_diff < 5e-6, f"Forward mismatch for h={h}"

        if backward:
            # Backward: sum of all outputs
            loss = y_local.sum()
            loss.backward()

            # Accumulate grads across ranks to form global x grad
            gx_local = x_local.grad if x_local.grad is not None else torch.zeros_like(x_local)
            gx_full = torch.zeros_like(x_full)
            # place local into correct slice
            gx_full[:, s:e, :] = gx_local
            dist.all_reduce(gx_full, op=dist.ReduceOp.SUM, group=group)

            # w grad should be the same across ranks; reduce sum equals single-rank grad
            gw_full = w.grad if w.grad is not None else torch.zeros_like(w)
            dist.all_reduce(gw_full, op=dist.ReduceOp.SUM, group=group)

            gx_ref = pack['gx'].to(device)
            gw_ref = pack['gw'].to(device)

            dx = (gx_full - gx_ref).abs()
            dw = (gw_full - gw_ref).abs()
            dx_max, dx_mean = float(dx.max().item()), float(dx.mean().item())
            dw_max, dw_mean = float(dw.max().item()), float(dw.mean().item())
            if rank == 0:
                print(f"h={h} backward parity: gx max {dx_max:.6f}, mean {dx_mean:.6f}; gw max {dw_max:.6f}, mean {dw_mean:.6f}")
            assert dx_max < 5e-5, f"Grad x mismatch for h={h}"
            assert dw_max < 5e-5, f"Grad w mismatch for h={h}"

            # Reset grads for next h
            x_local.grad = None

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'cp'], required=True)
    parser.add_argument('--cp_size', type=int, default=2)
    parser.add_argument('--save_ref', action='store_true')
    parser.add_argument('--varlen', action='store_true')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    if args.mode == 'single':
        run_single(save_ref=args.save_ref, varlen=args.varlen, backward=args.backward)
    else:
        run_cp(cp_size=args.cp_size, varlen=args.varlen, backward=args.backward)


if __name__ == '__main__':
    main()
