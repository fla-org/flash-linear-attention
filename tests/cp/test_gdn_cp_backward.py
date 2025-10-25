#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradient parity tests for GDN CP with short conv halo exchange.

Procedure:
- Build reference single-rank model, compute grad wrt embeddings on a short batch.
- Run CP with identical weights, shard inputs, compute grads.
- All-gather per-rank input-grads and compare to reference.
- Also validate parameter grads for q/k/v projections and short conv weights.

Run:
  python tests/cp/test_gdn_cp_backward.py --mode single --save_ref [--dtype fp32|bf16]
  torchrun --nproc_per_node=2 tests/cp/test_gdn_cp_backward.py --mode cp --cp_size 2 [--dtype fp32|bf16]
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


def _select_param(name: str) -> bool:
    """Filter for key projection and short conv weights."""
    keys = [
        'attn.q_proj.weight',
        'attn.k_proj.weight',
        'attn.v_proj.weight',
        'attn.q_conv1d',
        'attn.k_conv1d',
        'attn.v_conv1d',
    ]
    if any(k in name for k in keys) and name.endswith('weight'):
        return True
    return False


def _get_param_grads(model):
    return {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None and _select_param(n)}


def _to_dtype(model: torch.nn.Module, dtype_str: str) -> torch.dtype:
    if dtype_str == 'bf16':
        model.to(torch.bfloat16)
        return torch.bfloat16
    else:
        model.to(torch.float32)
        return torch.float32


def single(save_ref: bool, dtype_str: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = config_small()
    model = GatedDeltaNetForCausalLMCP(cfg).to(device)
    _to_dtype(model, dtype_str)

    torch.manual_seed(321)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 96), device=device)
    labels = torch.randint(0, cfg.vocab_size, (2, 96), device=device)

    model.train()
    model.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, labels=labels, cp_rank=0, cp_size=1, cp_group=None)
    out.loss.backward()

    # Grads
    emb_grad = model.get_input_embeddings().weight.grad.detach().cpu()
    param_grads = _get_param_grads(model)
    state = model.state_dict()

    ref = {'state': state, 'emb_grad': emb_grad, 'param_grads': param_grads, 'cfg': cfg, 'input_ids': input_ids.cpu(), 'labels': labels.cpu(), 'dtype': dtype_str}
    if save_ref:
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        torch.save(ref, REF_PATH)
        print(f"Saved reference grads to {REF_PATH}")
    return ref


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
    obj = [ref]
    dist.broadcast_object_list(obj, src=0, group=group)
    ref = obj[0]

    cfg = ref['cfg']
    dtype_str = ref.get('dtype', 'fp32')
    input_ids = ref['input_ids'].to(device)
    labels = ref['labels'].to(device)

    model = GatedDeltaNetForCausalLMCP(cfg).to(device)
    _to_dtype(model, dtype_str)
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
    print(f"Embedding grad max abs diff: {max_diff:.6f}, mean: {mean_diff:.6f}")
    assert max_diff < 5e-2, "Embedding grad mismatch beyond tolerance"

    # parameter grads parity (reduce-sum across ranks)
    param_grads_ref = {k: v.to(device) for k, v in ref['param_grads'].items()}
    tol_default = 5e-2
    tol_conv = 7e-2 if dtype_str == 'bf16' else tol_default
    for name, p in model.named_parameters():
        if not _select_param(name):
            continue
        if p.grad is None:
            raise AssertionError(f"Missing grad for {name}")
        g = p.grad.detach().clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=group)
        rg = param_grads_ref[name]
        d = (g.float() - rg.float()).abs()
        dmax, dmean = d.max().item(), d.mean().item()
        print(f"Grad {name} max {dmax:.6f}, mean {dmean:.6f}")
        tol = tol_conv if 'conv1d.weight' in name else tol_default
        assert dmax < tol, f"Param grad mismatch beyond tolerance for {name}"

    if rank == 0:
        print("✓ Backward CP parity (emb + key params) passed")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'cp'], required=True)
    parser.add_argument('--cp_size', type=int, default=2)
    parser.add_argument('--save_ref', action='store_true')
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    args = parser.parse_args()

    if args.mode == 'single':
        single(save_ref=args.save_ref, dtype_str=args.dtype)
    else:
        cp(cp_size=args.cp_size)


if __name__ == '__main__':
    main()
