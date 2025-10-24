#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test numerical correctness of Context Parallelism

This compares single-GPU results with CP results to ensure they match.

Usage:
    # Step 1: Generate reference on single GPU
    python test_numerical_correctness.py --mode single --save_ref
    
    # Step 2: Test with CP
    torchrun --nproc_per_node=2 test_numerical_correctness.py --mode cp --cp_size 2
"""

import argparse
import os
import torch
import torch.distributed as dist
from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet_cp import GatedDeltaNetForCausalLMCP


def setup_distributed(cp_size):
    """Setup distributed"""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    if world_size != cp_size:
        raise ValueError(f"world_size ({world_size}) must equal cp_size ({cp_size})")
    
    cp_group = dist.new_group(list(range(world_size)))
    return rank, device, cp_group


def create_model_and_data(config, device):
    """Create model and dummy data"""
    torch.manual_seed(42)
    
    model = GatedDeltaNetForCausalLMCP(config).to(device).to(torch.bfloat16)
    
    # Create dummy data
    batch_size = 2
    seq_len = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    return model, input_ids, labels


def test_single_gpu(save_ref=False):
    """Test on single GPU"""
    print("="*80)
    print("SINGLE GPU TEST")
    print("="*80)
    
    device = torch.device('cuda')
    
    config = GatedDeltaNetConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_heads=4,
        head_dim=64,
        vocab_size=1000,
        attn_mode='chunk',
        use_gate=True,
        use_short_conv=False,  # Disable for simplicity
    )
    
    model, input_ids, labels = create_model_and_data(config, device)
    
    print(f"\nConfig: hidden={config.hidden_size}, layers={config.num_hidden_layers}")
    print(f"Data: batch_size={input_ids.shape[0]}, seq_len={input_ids.shape[1]}")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            cp_rank=0,
            cp_size=1,
            cp_group=None
        )
    
    loss = outputs.loss
    logits = outputs.logits
    
    print(f"\nResults:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits mean: {logits.mean().item():.6f}")
    print(f"  Logits std: {logits.std().item():.6f}")
    
    if save_ref:
        print("\nSaving reference outputs...")
        torch.save({
            'config': config,
            'input_ids': input_ids.cpu(),
            'labels': labels.cpu(),
            'loss': loss.cpu(),
            'logits': logits.cpu(),
            'model_state': model.state_dict(),
        }, 'reference_single_gpu.pt')
        print("✓ Saved to reference_single_gpu.pt")
    
    print("="*80 + "\n")
    return loss, logits


def test_context_parallel(cp_size):
    """Test with context parallelism"""
    rank, device, cp_group = setup_distributed(cp_size)
    
    if rank == 0:
        print("="*80)
        print(f"CONTEXT PARALLELISM TEST (CP_SIZE={cp_size})")
        print("="*80)
    
    # Load reference
    if rank == 0:
        if not os.path.exists('reference_single_gpu.pt'):
            print("❌ ERROR: Run with --mode single --save_ref first!")
            dist.destroy_process_group()
            return
        
        ref = torch.load('reference_single_gpu.pt', weights_only=False)
        print("✓ Loaded reference")
        print(f"  Reference loss: {ref['loss'].item():.6f}")
    
    # Broadcast data
    if rank == 0:
        ref = torch.load('reference_single_gpu.pt', weights_only=False)
        config = ref['config']
        input_ids = ref['input_ids'].to(device).long()
        labels = ref['labels'].to(device).long()
        ref_loss = ref['loss'].to(device).float()
        ref_logits = ref['logits'].to(device).float()
    else:
        config = GatedDeltaNetConfig(
            hidden_size=256, num_hidden_layers=4, num_heads=4, head_dim=64,
            vocab_size=1000, attn_mode='chunk', use_gate=True, use_short_conv=False
        )
        input_ids = torch.empty((2, 512), dtype=torch.long, device=device)
        labels = torch.empty((2, 512), dtype=torch.long, device=device)
        ref_loss = torch.empty((), device=device, dtype=torch.float)
        ref_logits = torch.empty((2, 512, 1000), device=device, dtype=torch.float)

    # Broadcast
    dist.broadcast(input_ids, src=0, group=cp_group)
    dist.broadcast(labels, src=0, group=cp_group)
    dist.broadcast(ref_loss, src=0, group=cp_group)
    dist.broadcast(ref_logits, src=0, group=cp_group)
    
    print(f"[Rank {rank}] Data received")
    
    # --- build identical model on every rank ---
    print(f"[Rank {rank}] About to create model...")
    model = GatedDeltaNetForCausalLMCP(config).to(device).to(torch.bfloat16)
    print(f"[Rank {rank}] Model created successfully")

    # --- only rank 0 loads state, then broadcast to others ---
    if rank == 0:
        print(f"[Rank 0] Loading model state dict from file...")
        ref = torch.load('reference_single_gpu.pt', weights_only=False, map_location='cpu')
        model.load_state_dict(ref['model_state'])
        print(f"[Rank 0] Model state loaded")

    # broadcast all parameters (and buffers) from rank 0
    for p in model.state_dict().values():
        # ensure tensor on the right device & dtype before broadcast
        if isinstance(p, torch.Tensor):
            p_data = p.data.to(device)  # move to the local GPU
            dist.broadcast(p_data, src=0, group=cp_group)
            p.data.copy_(p_data)

    # (optional) a quick CUDA sync so NCCL uses the correct device queue
    torch.cuda.synchronize()

    # now barrier – everyone should arrive here fast
    print(f"[Rank {rank}] Waiting at barrier...")
    dist.barrier(group=cp_group)  # you can also do device_ids=[torch.cuda.current_device()]
    print(f"[Rank {rank}] Model weights synced")

        
    # Split sequence for CP
    _, seq_len = input_ids.shape
    chunk_size = seq_len // cp_size
    start = rank * chunk_size
    end = start + chunk_size
    
    input_ids_local = input_ids[:, start:end]
    labels_local = labels[:, start:end]
    
    print(f"[Rank {rank}] Processing chunk [{start}:{end}]")
    
    # IMPORTANT: Synchronize before forward pass
    dist.barrier(group=cp_group)
    print(f"[Rank {rank}] Starting forward pass...")
    
    # Forward pass with CP
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_local,
            labels=labels_local,
            cp_rank=rank,
            cp_size=cp_size,
            cp_group=cp_group
        )
    
    print(f"[Rank {rank}] Forward pass complete")
    
    loss_cp = outputs.loss
    logits_cp = outputs.logits
    
    # Gather logits from all ranks
    logits_list = [torch.zeros_like(logits_cp) for _ in range(cp_size)]
    dist.all_gather(logits_list, logits_cp, group=cp_group)
    logits_cp_full = torch.cat(logits_list, dim=1)
    
    # Compare on rank 0
    if rank == 0:
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        # Loss comparison
        loss_diff = abs(ref_loss.item() - loss_cp.item())
        loss_rel_diff = loss_diff / (abs(ref_loss.item()) + 1e-8)
        
        print(f"\n📊 Loss Comparison:")
        print(f"  Reference (single GPU): {ref_loss.item():.6f}")
        print(f"  CP (cp_size={cp_size}):   {loss_cp.item():.6f}")
        print(f"  Absolute difference:    {loss_diff:.6f}")
        print(f"  Relative difference:    {loss_rel_diff:.6%}")
        
        # Logits comparison
        logits_diff = (ref_logits - logits_cp_full).abs()
        max_diff = logits_diff.max().item()
        mean_diff = logits_diff.mean().item()
        
        # Better relative difference calculation
        # Use max of absolute values to avoid division by tiny numbers
        denominator = torch.maximum(ref_logits.abs(), logits_cp_full.abs()).clamp(min=1e-3)
        rel_diff = (logits_diff / denominator).max().item()
        
        # Also compute correlation
        ref_flat = ref_logits.flatten()
        cp_flat = logits_cp_full.flatten()
        correlation = torch.corrcoef(torch.stack([ref_flat, cp_flat]))[0, 1].item()
        
        print(f"\n📊 Logits Comparison:")
        print(f"  Max absolute diff:  {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")
        print(f"  Max relative diff:  {rel_diff:.2%} (improved calculation)")
        print(f"  Correlation:        {correlation:.6f} (1.0 = perfect match)")
        
        # Pass/Fail
        LOSS_TOLERANCE = 0.01  # 1%
        LOGITS_ABS_TOLERANCE = 0.02  # Allow 0.02 absolute difference
        LOGITS_REL_TOLERANCE = 0.10  # 10% with better calculation
        CORRELATION_THRESHOLD = 0.99  # High correlation required
        
        loss_pass = loss_rel_diff < LOSS_TOLERANCE
        logits_abs_pass = max_diff < LOGITS_ABS_TOLERANCE
        logits_rel_pass = rel_diff < LOGITS_REL_TOLERANCE
        correlation_pass = correlation > CORRELATION_THRESHOLD
        logits_pass = (logits_abs_pass or logits_rel_pass) and correlation_pass
        
        print(f"\n{'='*80}")
        print("TEST RESULTS")
        print(f"{'='*80}")
        print(f"  Loss test:        {'✅ PASS' if loss_pass else '❌ FAIL'} (threshold: {LOSS_TOLERANCE:.1%})")
        print(f"  Logits abs test:  {'✅ PASS' if logits_abs_pass else '❌ FAIL'} (threshold: {LOGITS_ABS_TOLERANCE})")
        print(f"  Logits rel test:  {'✅ PASS' if logits_rel_pass else '❌ FAIL'} (threshold: {LOGITS_REL_TOLERANCE:.1%})")
        print(f"  Correlation test: {'✅ PASS' if correlation_pass else '❌ FAIL'} (threshold: {CORRELATION_THRESHOLD})")
        print(f"  Overall logits:   {'✅ PASS' if logits_pass else '❌ FAIL'}")
        
        if loss_pass and logits_pass:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"   CP implementation is numerically correct!")
            print(f"   Loss difference: {loss_rel_diff:.2%}")
            print(f"   Max logits diff: {max_diff:.6f}")
            print(f"   Correlation: {correlation:.6f}")
        else:
            print(f"\n⚠️  SOME TESTS FAILED")
            if not loss_pass:
                print(f"   Loss difference too large: {loss_rel_diff:.2%} > {LOSS_TOLERANCE:.1%}")
            if not logits_pass:
                print(f"   Logits mismatch detected")
                print(f"   Consider: This may be acceptable numerical error")
        
        print("="*80 + "\n")
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Test CP numerical correctness')
    parser.add_argument('--mode', type=str, required=True, choices=['single', 'cp'],
                       help='Test mode')
    parser.add_argument('--cp_size', type=int, default=2, help='CP size for cp mode')
    parser.add_argument('--save_ref', action='store_true', help='Save reference (single mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        test_single_gpu(save_ref=args.save_ref)
    else:
        test_context_parallel(args.cp_size)


if __name__ == '__main__':
    main()