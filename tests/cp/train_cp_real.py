#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Realistic Training Script with Context Parallelism
- Each rank loads FULL sequences from disk
- Each rank processes its assigned CHUNK of the sequence
- Mimics real-world distributed training

Usage:
    torchrun --nproc_per_node=4 train_cp_real.py --cp_size=4
    torchrun --nproc_per_node=4 train_cp_real.py --cp_size=4 --seq_len=4096 --batch_size=1
"""

import argparse
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
# Import differentiable all_reduce
from torch.distributed.nn import all_reduce

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet_cp import GatedDeltaNetForCausalLMCP


def setup_distributed_cp(cp_size: int):
    """Setup distributed training with context parallelism."""
    dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size != cp_size:
        raise ValueError(
            f"For CP-only training, world_size ({world_size}) must equal cp_size ({cp_size}). "
            f"Launch with: torchrun --nproc_per_node={cp_size}"
        )
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Single CP group with all ranks
    cp_group = dist.new_group(list(range(world_size)))
    cp_rank = rank
    
    return rank, world_size, local_rank, device, cp_rank, cp_size, cp_group


class RealisticTextDataset(Dataset):
    """
    Realistic dataset that:
    1. Loads full sequences from storage
    2. Each worker/rank loads the SAME data (important for CP!)
    3. Returns full sequences (sharding happens in collate_fn)
    """
    
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        mode: str = 'random'
    ):
        """
        Args:
            num_samples: Number of training samples
            seq_len: Full sequence length (will be split across CP ranks)
            vocab_size: Vocabulary size
            mode: 'random' for synthetic data, 'file' for loading from disk
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mode = mode
        
        # For reproducibility: all ranks must see the same data!
        # In real scenarios, this would load from shared storage (NFS, S3, etc.)
        torch.manual_seed(42)
        
        print(f"    Dataset: {num_samples} samples, seq_len={seq_len}, vocab_size={vocab_size}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Load a full sequence.
        In real-world: this would load from disk, database, or cloud storage.
        All ranks load the SAME data.
        """
        # Use idx as seed for consistent data across ranks
        torch.manual_seed(42 + idx)
        
        # Simulate loading from storage
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        
        # Create labels (shifted by 1 for next-token prediction)
        labels = torch.cat([input_ids[1:], torch.tensor([self.vocab_size - 1])])
        
        # Attention mask (all 1s for now, could have padding in real scenarios)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sample_id': idx  # For debugging
        }


class CPCollator:
    """
    Custom collator that shards sequences across CP ranks.
    
    This is the KEY component that makes CP work in practice:
    - Receives FULL sequences from dataset
    - Shards them according to cp_rank
    - Returns only the chunk this rank should process
    """
    
    def __init__(self, cp_rank: int, cp_size: int, pad_token_id: int = 0):
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        """
        Collate and shard batch for CP.
        
        Args:
            batch: List of samples from dataset (full sequences)
        
        Returns:
            Dictionary with sharded tensors for this rank
        """
        # Stack into batch tensors (full sequences)
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        batch_size, seq_len = input_ids.shape
        
        # Ensure sequence length is divisible by cp_size
        if seq_len % self.cp_size != 0:
            # Pad to make divisible
            pad_len = self.cp_size - (seq_len % self.cp_size)
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
            seq_len = seq_len + pad_len
        
        # Shard: each rank gets its chunk
        chunk_size = seq_len // self.cp_size
        start_idx = self.cp_rank * chunk_size
        end_idx = start_idx + chunk_size
        
        # Extract chunk for this rank
        input_ids_chunk = input_ids[:, start_idx:end_idx]
        attention_mask_chunk = attention_mask[:, start_idx:end_idx]
        labels_chunk = labels[:, start_idx:end_idx]
        
        return {
            'input_ids': input_ids_chunk,
            'attention_mask': attention_mask_chunk,
            'labels': labels_chunk,
            'full_seq_len': seq_len,
            'chunk_start': start_idx,
            'chunk_end': end_idx
        }


def compute_loss_with_cp(
    model: GatedDeltaNetForCausalLMCP,
    batch: dict,
    cp_rank: int,
    cp_size: int,
    cp_group,
    device: torch.device
) -> torch.Tensor:
    """
    Compute loss with context parallelism.
    
    Important: Each rank computes loss only on its chunk.
    NO aggregation needed - gradients flow naturally through autograd!
    """
    # Move to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Debug: Check tensor shapes and values
    if torch.any(torch.isnan(input_ids.float())):
        print(f"Rank {cp_rank}: NaN detected in input_ids")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    try:
        # Forward pass (each rank processes its chunk)
        # Disable cu_seqlens for CP to avoid conflicts
        outputs = model(
            input_ids=input_ids,
            attention_mask=None,  # Disable attention_mask for now to avoid cu_seqlens
            labels=labels,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_group=cp_group
        )
        
        loss = outputs.loss
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"Rank {cp_rank}: NaN loss detected")
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return loss
        
    except Exception as e:
        print(f"Rank {cp_rank}: Error in forward pass: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)


def save_checkpoint(model, optimizer, step, checkpoint_dir, rank):
    """Save checkpoint (only on rank 0)"""
    if rank != 0:
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    print(f"✓ Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint"""
    if not os.path.exists(checkpoint_path):
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    
    print(f"✓ Loaded checkpoint from step {step}")
    return step


def main():
    parser = argparse.ArgumentParser(description='Realistic CP Training')
    
    # Distributed
    parser.add_argument('--cp_size', type=int, required=True, help='Context parallel size')
    
    # Data
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--seq_len', type=int, default=2048, help='Full sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (same on all ranks)')
    
    # Training
    parser.add_argument('--num_steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N steps')
    
    # Model
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank, device, cp_rank, cp_size, cp_group = setup_distributed_cp(args.cp_size)
    
    if rank == 0:
        print("\n" + "="*80)
        print("REALISTIC CONTEXT PARALLELISM TRAINING")
        print("="*80)
        print(f"\n🔧 Setup:")
        print(f"  World size: {world_size}")
        print(f"  CP size: {cp_size}")
        print(f"  CP rank: {cp_rank}")
        print(f"  Device: {device}")
    
    # Create model configuration
    config = GatedDeltaNetConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        vocab_size=args.vocab_size,
        attn_mode='chunk',
        use_gate=True,
        use_short_conv=False, # Simplify for testing
        expand_v=1.0,
    )
    
    if rank == 0:
        print(f"\n📐 Model Configuration:")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_layers: {config.num_hidden_layers}")
        print(f"  num_heads: {config.num_heads}")
        print(f"  head_dim: {config.head_dim}")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"  Parameters: ~{sum(p.numel() for p in GatedDeltaNetForCausalLMCP(config).parameters()):,}")
    
    # Create model
    model = GatedDeltaNetForCausalLMCP(config).to(device)
    model = model.to(torch.bfloat16)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, args.resume_from, device)
    
    # Create dataset (all ranks load the same data!)
    if rank == 0:
        print(f"\n📚 Dataset:")
        print(f"  Samples: {args.num_samples}")
        print(f"  Full sequence length: {args.seq_len}")
        print(f"  Chunk per rank: {args.seq_len // args.cp_size}")
        print(f"  Batch size: {args.batch_size}")
    
    dataset = RealisticTextDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        mode='random'
    )
    
    # Create dataloader with custom collator
    # Key: All ranks see the same data, but collator shards it
    collator = CPCollator(cp_rank=cp_rank, cp_size=cp_size, pad_token_id=0)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # No shuffle to keep data consistent across ranks for CP (may need shared deterministic sampler)
        collate_fn=collator,  # Shard sequences in collator
        num_workers=0,  # Keep simple for now
        pin_memory=True
    )
    
    if rank == 0:
        print(f"\n🚀 Training:")
        print(f"  Total steps: {args.num_steps}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Checkpoint every: {args.save_every} steps")
        print("\n" + "="*80 + "\n")
    
    # Training loop
    model.train()
    total_loss = 0.0
    step = start_step
    optimizer.zero_grad()
    
    while step < args.num_steps:
        for batch in dataloader:
            if step >= args.num_steps:
                break
            
            # Debug: print chunk info (only rank 0, only first batch)
            if rank == 0 and step == start_step:
                print(f"📦 Batch info:")
                print(f"  input_ids shape: {batch['input_ids'].shape}")
                print(f"  Chunk range: [{batch['chunk_start']}:{batch['chunk_end']}]")
                print(f"  Full sequence length: {batch['full_seq_len']}\n")
            
            # Forward and backward
            loss = compute_loss_with_cp(
                model=model,
                batch=batch,
                cp_rank=cp_rank,
                cp_size=cp_size,
                cp_group=cp_group,
                device=device
            )
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
            
            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Logging (rank 0 only)
                if rank == 0 and (step + 1) % 10 == 0:
                    avg_loss = total_loss * args.gradient_accumulation_steps / 10
                    print(f"Step {step + 1}/{args.num_steps} | Loss: {avg_loss:.4f}")
                    total_loss = 0.0
            
            # Save checkpoint
            if rank == 0 and (step + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, step + 1, args.checkpoint_dir, rank)
            
            step += 1
    
    # Final checkpoint
    if rank == 0:
        print("\n" + "="*80)
        print("Training complete!")
        save_checkpoint(model, optimizer, step, args.checkpoint_dir, rank)
        print(f"Final model saved to {args.checkpoint_dir}")
        print("="*80 + "\n")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    main()