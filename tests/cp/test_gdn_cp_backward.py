#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradient parity tests for GDN CP with short conv halo exchange.

Procedure:
- Build reference single-rank model, compute grad wrt embeddings on a short batch.
- Run CP with identical weights, shard inputs, compute grads.
- All-gather per-rank input-grads and compare to reference.

Run:
  python tests/cp/test_gdn_cp_backward.py --mode single --save_ref
  torchrun --nproc_per_node=2 tests/cp/test_gdn_cp_backward.py --mode cp --cp_size 2
"""

import argparse
import os
import torch
import torch.distributed as dist

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet_cp import GatedDeltaNetForCausalLMCP

REF_PATH = 'tests/cp/.gdn_cp_backward_ref.pt'

def config_small():
    return GatedDeltaNetConfig(
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


def single(save_ref: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = config_small()
    model = GatedDeltaNetForCausalLMCP(cfg).to(device).to(torch.bfloat16)

    torch.manual_seed(321)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 96), device=device)
    labels = torch.randint(0, cfg.vocab_size, (2, 96), device=device)

    model.train()
    model.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, labels=labels, cp_rank=0, cp_size=1, cp_group=None)
    out.loss.backward()

    # Grab grad wrt input embeddings table
    emb_grad = model.get_input_embeddings().weight.grad.detach().cpu()
    state = model.state_dict()

    if save_ref:
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        torch.save({'state': state, 'emb_grad': emb_grad, 'cfg': cfg, 'input_ids': input_ids.cpu(), 'labels': labels.cpu()}, REF_PATH)
        print(f"Saved reference grads to {REF_PATH}")
    return {'state': state, 'emb_grad': emb_grad, 'cfg': cfg, 'input_ids': input_ids.cpu(), 'labels': labels.cpu()}


def setup(cp_size: int):
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(cp_size)))
    return rank, device, group


def cp(cp_size: int):
    rank, device, group = setup(cp_size)

    # Load ref
    if rank == 0:
        assert os.path.exists(REF_PATH), f"Run single first to create {REF_PATH}"
        ref = torch.load(REF_PATH, weights_only=False, map_location='cpu')
    else:
        ref = None
    ref = dist.broadcast_object_list([ref], src=0)[0] if hasattr(dist, 'broadcast_object_list') else ref

    cfg = ref['cfg']
    input_ids = ref['input_ids'].to(device)
    labels = ref['labels'].to(device)

    model = GatedDeltaNetForCausalLMCP(cfg).to(device).to(torch.bfloat16)
    model.load_state_dict(ref['state'])

    # shard
    B, T = input_ids.shape
    assert T % cp_size == 0
    chunk = T // cp_size
    s = rank * chunk
    e = s + chunk
    in_local = input_ids[:, s:e]
    lab_local = labels[:, s:e]

    # train mode grad test
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(input_ids=in_local, labels=lab_local, cp_rank=rank, cp_size=cp_size, cp_group=group)
    out.loss.backward()

    # collect embedding grads and sum across ranks (they should match full grads)
    emb_grad_local = model.get_input_embeddings().weight.grad
    emb_grad_full = emb_grad_local.clone()
    dist.all_reduce(emb_grad_full, op=dist.ReduceOp.SUM, group=group)

    ref_grad = ref['emb_grad'].to(device)
    diff = (emb_grad_full.float() - ref_grad.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"Grad max abs diff: {max_diff:.6f}, mean: {mean_diff:.6f}")
    assert max_diff < 5e-2, "Embedding grad mismatch beyond tolerance"
    if rank == 0:
        print("✓ Backward CP parity passed")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'cp'], required=True)
    parser.add_argument('--cp_size', type=int, default=2)
    parser.add_argument('--save_ref', action='store_true')
    args = parser.parse_args()

    if args.mode == 'single':
        single(save_ref=args.save_ref)
    else:
        cp(cp_size=args.cp_size)


if __name__ == '__main__':
    main()
