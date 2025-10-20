#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark memory usage with different CP configurations

Usage:
    # Single GPU
    python benchmark_memory.py --cp_size 1
    
    # With CP
    torchrun --nproc_per_node=4 benchmark_memory.py --cp_size 4
"""

import argparse
import os
import torch
import torch.distributed as dist
from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet_cp import GatedDeltaNetForCausalLMCP


def setup_distributed(cp_size):
    """Setup distributed"""
    if cp_size > 1:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        cp_group = dist.new_group(list(range(cp_size)))
    else:
        rank = 0
        device = torch.device('cuda')
        cp_group = None
    
    return rank, device, cp_group


def get_memory_stats():
    """Get current GPU memory usage"""
    allocated = torch.cuda.memory_allocated() / 1e9  # GB
    reserved = torch.cuda.memory_reserved() / 1e9    # GB
    return allocated, reserved


def benchmark_config(rank, device, cp_group, cp_size, seq_len, batch_size, hidden_size, num_layers):
    """Benchmark a single configuration"""
    # Reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Config
    config = GatedDeltaNetConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_heads=hidden_size // 64,
        head_dim=64,
        vocab_size=32000,
        attn_mode='chunk',
        use_gate=True,
        use_short_conv=False,
    )
    
    # Create model
    model = GatedDeltaNetForCausalLMCP(config).to(device).to(torch.bfloat16)
    
    # Initial memory
    mem_model, _ = get_memory_stats()
    
    # Calculate chunk size for this rank
    chunk_size = seq_len // cp_size
    
    # Create data (each rank gets its chunk)
    input_ids = torch.randint(0, 32000, (batch_size, chunk_size), device=device)
    labels = torch.randint(0, 32000, (batch_size, chunk_size), device=device)
    
    mem_data, _ = get_memory_stats()
    
    # Forward pass
    try:
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            cp_rank=rank,
            cp_size=cp_size,
            cp_group=cp_group
        )
        loss = outputs.loss
        
        mem_forward, _ = get_memory_stats()
        
        # Backward pass
        loss.backward()
        
        mem_backward, _ = get_memory_stats()
        
        # Peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        
        success = True
        error = None
        
    except RuntimeError as e:
        success = False
        error = str(e)
        mem_forward = 0
        mem_backward = 0
        peak_memory = 0
    
    # Clean up
    del model, input_ids, labels
    if success:
        del outputs, loss
    torch.cuda.empty_cache()
    
    # Return results (only from rank 0)
    if rank == 0:
        return {
            'success': success,
            'error': error,
            'mem_model': mem_model,
            'mem_data': mem_data - mem_model,
            'mem_forward': mem_forward - mem_data,
            'mem_backward': mem_backward - mem_forward,
            'peak_memory': peak_memory,
            'total_seq_len': seq_len,
            'chunk_size': chunk_size,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark memory usage')
    parser.add_argument('--cp_size', type=int, default=1, help='CP size')
    parser.add_argument('--seq_lens', type=str, default='1024,2048,4096,8192,16384', help='Comma-separated sequence lengths')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    
    args = parser.parse_args()
    
    # Parse sequence lengths
    seq_lens = [int(x) for x in args.seq_lens.split(',')]
    
    # Setup distributed once at the beginning
    rank, device, cp_group = setup_distributed(args.cp_size)
    
    if rank == 0:
        print("="*100)
        print(f"MEMORY BENCHMARK: CP_SIZE={args.cp_size}")
        print("="*100)
        print(f"\nConfiguration:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Hidden size: {args.hidden_size}")
        print(f"  Num layers: {args.num_layers}")
        print(f"  Sequence lengths: {seq_lens}")
        print()
    
    results = []
    
    for seq_len in seq_lens:
        if rank == 0:
            print(f"\nTesting seq_len={seq_len}...")
        
        result = benchmark_config(
            rank=rank,
            device=device,
            cp_group=cp_group,
            cp_size=args.cp_size,
            seq_len=seq_len,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )
        
        if rank == 0 and result:
            results.append(result)
            
            if result['success']:
                print(f"  ✓ Success")
                print(f"    Model:    {result['mem_model']:.2f} GB")
                print(f"    Data:     {result['mem_data']:.2f} GB")
                print(f"    Forward:  {result['mem_forward']:.2f} GB")
                print(f"    Backward: {result['mem_backward']:.2f} GB")
                print(f"    Peak:     {result['peak_memory']:.2f} GB")
                print(f"    Chunk/rank: {result['chunk_size']} tokens")
            else:
                print(f"  ❌ Failed: {result['error'][:100]}")
    
    # Summary
    if rank == 0:
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        print(f"\n{'Seq Len':<10} {'Chunk':<10} {'Model':<10} {'Peak':<10} {'Status':<10}")
        print("-"*50)
        
        for result in results:
            status = "✓" if result['success'] else "✗"
            print(f"{result['total_seq_len']:<10} {result['chunk_size']:<10} "
                  f"{result['mem_model']:.2f} GB    {result['peak_memory']:.2f} GB    {status}")
        
        print("\n" + "="*100)
        print("\nTo compare with different CP sizes, run:")
        print(f"  python benchmark_memory.py --cp_size 1")
        print(f"  torchrun --nproc_per_node=2 benchmark_memory.py --cp_size 2")
        print(f"  torchrun --nproc_per_node=4 benchmark_memory.py --cp_size 4")
        print()
    
    # Clean up distributed at the end
    if args.cp_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()