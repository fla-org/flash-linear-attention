#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deadlock/stability smoke test for CP + short conv halo.

Run:
  torchrun --nproc_per_node=2 tests/cp/test_cp_stability.py --cp_size 2
"""

import argparse
import os

import torch
import torch.distributed as dist

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet_cp import GatedDeltaNetForCausalLMCP


def setup(cp_size: int):
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(cp_size)))
    return rank, device, group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp_size', type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        if int(os.environ.get('RANK', '0')) == 0:
            print('Skipping stability test: CUDA not available')
        return

    rank, device, group = setup(args.cp_size)

    cfg = GatedDeltaNetConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=4,
        head_dim=32,
        vocab_size=1000,
        attn_mode='chunk',
        use_gate=True,
        use_short_conv=True,
        conv_size=4,
        expand_v=1.0,
    )
    model = GatedDeltaNetForCausalLMCP(cfg).to(device).to(torch.bfloat16)

    B = 2
    T = 256
    assert T % args.cp_size == 0
    chunk = T // args.cp_size
    s = rank * chunk
    e = s + chunk

    torch.manual_seed(777)
    input_full = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    labels_full = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    # Run multiple iterations with barriers to catch potential hangs
    for it in range(5):
        dist.barrier(group=group)
        input_local = input_full[:, s:e].contiguous()
        labels_local = labels_full[:, s:e].contiguous()
        out = model(input_ids=input_local, labels=labels_local,
                    cp_rank=rank, cp_size=args.cp_size, cp_group=group)
        loss_val = out.loss.item()
        if rank == 0:
            print(f"Iter {it}: loss={loss_val:.4f}")

    if rank == 0:
        print('✓ CP stability smoke test passed (no deadlocks)')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
