#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradient parity tests for GDN CP with short conv halo exchange.

Procedure:
- Build reference single-rank model, compute grad wrt embeddings on a short batch.
- Run CP with identical weights, shard inputs, compute grads.
- All-reduce(AVG) per-rank grads and compare to reference.
- Also validate parameter grads for q/k/v projections and short conv weights.

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


def _set_determinism(seed: int = 321):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic GEMMs on Ampere+
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    return {
        n: p.grad.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.grad is not None and _select_param(n)
    }


def single(save_ref: bool):
    _set_determinism(321)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = config_small()
    model = GatedDeltaNetForCausalLMCP(cfg).to(device).to(torch.bfloat16)

    # fixed data for reproducibility
    torch.manual_seed(321)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 96), device=device)
    labels = torch.randint(0, cfg.vocab_size, (2, 96), device=device)

    model.train()
    model.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, labels=labels, cp_rank=0, cp_size=1, cp_group=None)
    out.loss.backward()

    # Grads + state
    emb_grad = model.get_input_embeddings().weight.grad.detach().cpu()
    param_grads = _get_param_grads(model)
    state = model.state_dict()

    ref = {
        'state': state,
        'emb_grad': emb_grad,
        'param_grads': param_grads,
        'cfg': cfg,
        'input_ids': input_ids.cpu(),
        'labels': labels.cpu()
    }
    if save_ref:
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        torch.save(ref, REF_PATH)
        print(f"Saved reference grads to {REF_PATH}")
    return ref


def setup(cp_size: int):
    # env init via torchrun
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()
    assert world_size == cp_size, f"world_size ({world_size}) must equal cp_size ({cp_size}) for this test"
    rank = dist.get_rank()

    # require LOCAL_RANK to be present when using GPUs
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device('cpu')

    group = dist.group.WORLD  # use the full world as the CP group
    return rank, device, group


def cp(cp_size: int):
    _set_determinism(321)
    rank, device, group = setup(cp_size)

    # Load ref on every rank (avoids large object broadcast and stays simple)
    assert os.path.exists(REF_PATH), f"Run single first to create {REF_PATH}"
    ref = torch.load(REF_PATH, weights_only=False, map_location='cpu')

    cfg = ref['cfg']
    input_ids = ref['input_ids'].to(device)
    labels = ref['labels'].to(device)

    model = GatedDeltaNetForCausalLMCP(cfg).to(device).to(torch.bfloat16)
    model.load_state_dict(ref['state'])

    # shard sequence dimension equally
    B, T = input_ids.shape
    assert T % cp_size == 0, f"T ({T}) must be divisible by cp_size ({cp_size})"
    chunk = T // cp_size
    s = dist.get_rank() * chunk
    e = s + chunk
    in_local = input_ids[:, s:e]
    lab_local = labels[:, s:e]

    # train mode grad test
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(input_ids=in_local, labels=lab_local,
                cp_rank=dist.get_rank(), cp_size=cp_size, cp_group=group)
    out.loss.backward()

    # ===== Embedding grad parity =====
    emb_grad_local = model.get_input_embeddings().weight.grad
    emb_grad_avg = emb_grad_local.clone()
    dist.all_reduce(emb_grad_avg, op=dist.ReduceOp.AVG, group=group)  # average to match single-rank scale

    ref_grad = ref['emb_grad'].to(device)
    d = (emb_grad_avg.float() - ref_grad.float()).abs()
    dmax, dmean = d.max().item(), d.mean().item()

    # tighter tol with determinism + bf16
    emb_atol, emb_rtol = 2e-2, 1e-2
    ok_emb = torch.allclose(emb_grad_avg.float(), ref_grad.float(), atol=emb_atol, rtol=emb_rtol)
    print(f"[rank {dist.get_rank()}] Embedding grad: max {dmax:.6f}, mean {dmean:.6f}, allclose={ok_emb}")
    if not ok_emb:
        raise AssertionError(f"Embedding grad mismatch: max {dmax:.6f}, mean {dmean:.6f} (atol={emb_atol}, rtol={emb_rtol})")

    # ===== Parameter grad parity (q/k/v linear + short conv) =====
    param_grads_ref = {k: v.to(device) for k, v in ref['param_grads'].items()}
    lin_atol, lin_rtol = 3e-2, 1e-2
    conv_atol, conv_rtol = 5e-2, 1e-2

    for name, p in model.named_parameters():
        if not _select_param(name):
            continue
        if p.grad is None:
            raise AssertionError(f"Missing grad for {name}")

        g = p.grad.detach().clone()
        dist.all_reduce(g, op=dist.ReduceOp.AVG, group=group)  # average to match single-rank scale

        rg = param_grads_ref[name]
        diff = (g.float() - rg.float()).abs()
        dmax, dmean = diff.max().item(), diff.mean().item()

        if 'conv1d.weight' in name:
            atol, rtol = conv_atol, conv_rtol
        else:
            atol, rtol = lin_atol, lin_rtol

        ok = torch.allclose(g.float(), rg.float(), atol=atol, rtol=rtol)
        print(f"[rank {dist.get_rank()}] Grad {name}: max {dmax:.6f}, mean {dmean:.6f}, allclose={ok}")
        if not ok:
            raise AssertionError(f"Param grad mismatch for {name}: max {dmax:.6f}, mean {dmean:.6f} (atol={atol}, rtol={rtol})")

    dist.barrier(group=group)
    if dist.get_rank() == 0:
        print("✓ Backward CP parity (emb + key params) passed")

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
