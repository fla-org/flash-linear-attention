#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Forward parity tests for GDN CP with short conv halo exchange.

This compares single-rank outputs to multi-rank CP outputs by sharding inputs
and stitching logits back. Covers equal-length and varlen cases.

Run equal-length:
  python tests/cp/test_gdn_cp_forward.py --mode single --save_ref
  torchrun --nproc_per_node=2 tests/cp/test_gdn_cp_forward.py --mode cp --cp_size 2

Run varlen:
  python tests/cp/test_gdn_cp_forward.py --mode single --save_ref --varlen
  torchrun --nproc_per_node=2 tests/cp/test_gdn_cp_forward.py --mode cp --cp_size 2 --varlen
"""

import argparse
import os
from typing import Optional

import torch
import torch.distributed as dist

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet_cp import GatedDeltaNetForCausalLMCP


REF_PATH = 'tests/cp/.gdn_cp_forward_ref.pt'


def create_config(use_short_conv: bool = True) -> GatedDeltaNetConfig:
    return GatedDeltaNetConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_heads=4,
        head_dim=64,
        vocab_size=1000,
        attn_mode='chunk',
        use_gate=True,
        use_short_conv=use_short_conv,
        conv_size=4,
        expand_v=1.0,
    )


def create_data(batch_size: int, seq_len: int, vocab_size: int, varlen: bool = False):
    torch.manual_seed(123)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    cu_seqlens = None
    if varlen:
        # Create two sequences per batch with a boundary at seq_len//2
        # We flatten by layer anyway; here we just pass cu_seqlens via kwargs when needed
        cu_seqlens = torch.tensor([0, seq_len // 2, seq_len], dtype=torch.long)
    return input_ids, labels, cu_seqlens


def run_single(save_ref: bool, varlen: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = create_config(use_short_conv=True)
    model = GatedDeltaNetForCausalLMCP(config).to(device).to(torch.bfloat16)

    input_ids, labels, cu_seqlens = create_data(batch_size=2, seq_len=128, vocab_size=config.vocab_size, varlen=varlen)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            labels=labels,
            cp_rank=0, cp_size=1, cp_group=None,
            cu_seqlens=cu_seqlens.to(device) if cu_seqlens is not None else None,
            cp_shard_start_idx=0 if varlen else None,
        )
    ref = {
        'state': model.state_dict(),
        'loss': out.loss.detach().cpu(),
        'logits': out.logits.detach().cpu(),
        'config': config,
        'input_ids': input_ids.cpu(),
        'labels': labels.cpu(),
        'cu_seqlens': cu_seqlens,
    }
    if save_ref:
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        torch.save(ref, REF_PATH)
        print(f"Saved reference to {REF_PATH}")
    return ref


def setup_dist(cp_size: int):
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(cp_size)))
    return rank, device, group


def run_cp(cp_size: int, varlen: bool):
    rank, device, group = setup_dist(cp_size)

    # Load and broadcast reference
    if rank == 0:
        assert os.path.exists(REF_PATH), f"Run single first to create {REF_PATH}"
        ref = torch.load(REF_PATH, weights_only=False, map_location='cpu')
    else:
        ref = None
    obj_list = [ref]
    dist.broadcast_object_list(obj_list, src=0, group=group)
    ref = obj_list[0]

    config: GatedDeltaNetConfig = ref['config']
    input_ids = ref['input_ids'].to(device)
    labels = ref['labels'].to(device)
    cu_seqlens = ref['cu_seqlens']

    model = GatedDeltaNetForCausalLMCP(config).to(device).to(torch.bfloat16)
    model.load_state_dict(ref['state'])

    # Shard inputs
    B, T = input_ids.shape
    assert T % cp_size == 0
    chunk = T // cp_size
    s = rank * chunk
    e = s + chunk

    input_local = input_ids[:, s:e]
    labels_local = labels[:, s:e]

    cp_shard_start_idx = s

    with torch.no_grad():
        out = model(
            input_ids=input_local,
            labels=labels_local,
            cp_rank=rank, cp_size=cp_size, cp_group=group,
            cu_seqlens=cu_seqlens.to(device) if cu_seqlens is not None else None,
            # Only needed for varlen to avoid crossing seq boundaries in halo exchange
            cp_shard_start_idx=cp_shard_start_idx if cu_seqlens is not None else None,
        )

    # Gather logits across ranks (sequence dimension)
    logits_local = out.logits  # [B, chunk, V]
    logits_list = [torch.zeros_like(logits_local) for _ in range(cp_size)]
    dist.all_gather(logits_list, logits_local, group=group)
    logits_full = torch.cat(logits_list, dim=1)

    # Compare to reference on rank 0
    if rank == 0:
        ref_logits = ref['logits'].to(device)
        diff = (logits_full.float() - ref_logits.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"Max abs diff: {max_diff:.6f}, mean abs diff: {mean_diff:.6f}")
        assert max_diff < 5e-2, "Logits mismatch beyond tolerance"
        print("✓ Forward CP parity passed")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'cp'], required=True)
    parser.add_argument('--cp_size', type=int, default=2)
    parser.add_argument('--save_ref', action='store_true')
    parser.add_argument('--varlen', action='store_true')
    args = parser.parse_args()

    if args.mode == 'single':
        run_single(save_ref=args.save_ref, varlen=args.varlen)
    else:
        run_cp(cp_size=args.cp_size, varlen=args.varlen)


if __name__ == '__main__':
    main()
