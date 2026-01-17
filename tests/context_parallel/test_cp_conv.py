"""
Test for Context Parallel (CP) Causal Convolution 1D

Context Parallel Principle for Causal Conv1d:
=============================================

Causal convolution has a dependency on previous tokens due to the sliding window.
In a standard implementation, each rank processes the full sequence sequentially.

With Context Parallel:
1. Sequence Partitioning: The input sequence is split across ranks along the sequence dimension.
   - Rank 0: tokens [0, T/N)
   - Rank 1: tokens [T/N, 2T/N)
   - Rank 2: tokens [2T/N, 3T/N)
   - ...

2. Forward Pass:
   - Each rank processes its local chunk independently
   - Non-first ranks need the last (W-1) tokens from the previous rank as initial_state
   - Communication: Previous rank sends its tail tokens (last W-1 tokens) to current rank
   - Current rank receives and constructs initial_state from previous rank's tail
   - This allows parallel computation while maintaining causal dependencies

3. Backward Pass:
   - Gradients need to be corrected because tokens used as initial_state by next rank
     also contribute to gradients
   - Communication: Current rank sends d_initial_state to previous rank
   - Previous rank adds received gradients to its tail tokens (last W-1 tokens)
   - This ensures gradient correctness across rank boundaries

Key Insight:
- The last (W-1) tokens of each rank are used as initial_state by the next rank
- These tokens need gradient contributions from both local computation and next rank
- Communication overhead is minimal: only (W-1) tokens per rank boundary

Example with kernel_size W=4:
- Rank 0 processes tokens [0, T/2), sends last 3 tokens to Rank 1
- Rank 1 receives 3 tokens, uses as initial_state, processes tokens [T/2, T)
- In backward: Rank 1 sends d_initial_state to Rank 0, Rank 0 adds to its last 3 tokens
"""

import os

import pytest
import torch
import torch.distributed as dist

from fla.modules.convolution import causal_conv1d
from fla.ops.cp import get_cp_context, set_cp_context
from fla.utils import device


def setup_distributed():
    """Initialize distributed environment."""
    if 'RANK' not in os.environ:
        return False

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True


def cleanup_distributed():
    """Clean up distributed environment."""
    dist.destroy_process_group()


def assert_close(name, ref, tri, atol=1e-3, rtol=1e-3):
    """Helper function for assertion with detailed diff output."""
    diff = (ref - tri).abs().max().item()
    print(f"  Checking {name:<10} | Max Diff: {diff:.6f}")
    assert diff < atol, f"{name} mismatch. Max diff: {diff}"


def run_cp_test(dtype=torch.float32):
    """
    Run Context Parallel causal convolution test.

    This test verifies that CP implementation produces identical results
    to the standard sequential implementation.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # -------------------------------------------------------
    # Step 1: Prepare Global Data
    # All ranks generate the same data using the same seed
    # -------------------------------------------------------
    torch.manual_seed(42)
    B, T, D, W = 1, 1024, 128, 4  # B must be 1 for varlen mode

    # Global input and gradient tensors
    x_global = torch.randn(B, T, D, device=device, dtype=dtype)
    dy_global = torch.randn(B, T, D, device=device, dtype=dtype)

    # Weight and bias parameters
    weight = torch.randn(D, W, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype)

    # Broadcast weights to ensure consistency across ranks
    dist.broadcast(weight, src=0)
    dist.broadcast(bias, src=0)

    # Construct variable-length sequence boundaries (across rank boundaries)
    lengths = [300, 400, T - 700]
    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32)

    activation = 'swish'

    # -------------------------------------------------------
    # Step 2: Reference Run (Sequential Implementation)
    # Compute on Rank 0 only as the ground truth baseline
    # -------------------------------------------------------
    ref_out, ref_dx, ref_dw, ref_db = None, None, None, None

    if rank == 0:
        x_ref = x_global.clone().detach().requires_grad_(True)
        weight_ref = weight.clone().detach().requires_grad_(True)
        bias_ref = bias.clone().detach().requires_grad_(True)

        # Standard forward pass (using triton backend)
        y_ref, _ = causal_conv1d(
            x=x_ref,
            weight=weight_ref,
            bias=bias_ref,
            activation=activation,
            backend='triton',
            cu_seqlens=cu_seqlens_global,
        )

        # Standard backward pass
        y_ref.backward(dy_global)

        ref_out = y_ref.detach()
        ref_dx = x_ref.grad.detach()
        ref_dw = weight_ref.grad.detach()
        ref_db = bias_ref.grad.detach()

    # -------------------------------------------------------
    # Step 3: Context Parallel Run
    # All ranks process their chunks in parallel
    # -------------------------------------------------------
    dist.barrier()

    # A. Setup CP Context
    # This configures the context parallel environment with sequence boundaries
    # and kernel size information needed for communication
    set_cp_context(cu_seqlens_global, group=dist.group.WORLD, kernel_size=W)
    context = get_cp_context()

    # B. Data Partitioning (Slice Input)
    # Each rank gets a contiguous chunk of the sequence
    # Rank 0: [0, T/N), Rank 1: [T/N, 2T/N), etc.
    chunk_size = T // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size

    x_local = x_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    dy_local = dy_global[:, start_idx:end_idx, :].clone()
    weight_local = weight.clone().detach().requires_grad_(True)
    bias_local = bias.clone().detach().requires_grad_(True)

    # Debug: Print context information
    print(f"[Rank {rank}] cu_seqlens_global: {cu_seqlens_global.tolist()}")
    print(f"[Rank {rank}] context.cu_seqlens: {context.cu_seqlens.tolist()}")
    print(f"[Rank {rank}] pre_num_ranks: {context.pre_num_ranks}")
    print(f"[Rank {rank}] pre_num_conv_tokens: {context.pre_num_conv_tokens}")
    print(f"[Rank {rank}] start_idx: {start_idx}, end_idx: {end_idx}")
    print(f"[Rank {rank}] x_local shape: {x_local.shape}")
    dist.barrier()

    # C. CP Forward Pass
    # Key: Pass cp_context to enable context parallel mode
    # - Non-first ranks will receive initial_state from previous rank
    # - Communication happens inside causal_conv1d when cp_context is provided
    y_local, _ = causal_conv1d(
        x=x_local,
        weight=weight_local,
        bias=bias_local,
        activation=activation,
        cu_seqlens=context.cu_seqlens,
        cp_context=context,
    )

    # Debug: Compare forward output
    dist.barrier()
    if rank == 0:
        # Rank 0's output should match ref_out[:, :chunk_size, :]
        ref_local = ref_out[:, :chunk_size, :]
        diff = (y_local - ref_local).abs().max().item()
        print(f"[Rank 0] Forward diff (local vs ref[:chunk_size]): {diff:.6f}")
    dist.barrier()

    # D. CP Backward Pass
    # Gradients are automatically corrected through communication
    # - Current rank sends d_initial_state to previous rank
    # - Previous rank adds received gradients to its tail tokens
    y_local.backward(dy_local)

    # -------------------------------------------------------
    # Step 4: Result Aggregation and Verification
    # -------------------------------------------------------

    # Aggregate Output: Concatenate outputs from all ranks
    y_gathered = [torch.zeros_like(y_local) for _ in range(world_size)]
    dist.all_gather(y_gathered, y_local)
    y_cp_global = torch.cat(y_gathered, dim=1)

    # Aggregate Input Gradients (dx): Concatenate gradients from all ranks
    dx_gathered = [torch.zeros_like(x_local.grad) for _ in range(world_size)]
    dist.all_gather(dx_gathered, x_local.grad)
    dx_cp_global = torch.cat(dx_gathered, dim=1)

    # Aggregate Weight Gradients (dw, db): Sum across all ranks
    # Since weights are shared, gradients are summed (not concatenated)
    dw_cp = weight_local.grad.clone()
    db_cp = bias_local.grad.clone()
    dist.all_reduce(dw_cp, op=dist.ReduceOp.SUM)
    dist.all_reduce(db_cp, op=dist.ReduceOp.SUM)

    # Verification: Compare CP results with reference (only on Rank 0)
    if rank == 0:
        print(f"\n[CP Conv1d] Verification Results (CP Size={world_size}):")

        # 1. Check Output
        assert_close("Output", ref_out, y_cp_global, atol=1e-3)

        # 2. Check Input Gradients (dx)
        assert_close("dx (Input)", ref_dx, dx_cp_global, atol=1e-3)

        # 3. Check Weight Gradients (dw)
        assert_close("dw (Weight)", ref_dw, dw_cp, atol=1e-3)

        # 4. Check Bias Gradients (db)
        assert_close("db (Bias)", ref_db, db_cp, atol=1e-3)

        print("✅ [CP Conv1d] Test Passed!\n")

    dist.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="At least 2 GPUs required")
def test_cp_convolution():
    """
    Test Context Parallel causal convolution.

    This test must be run with torchrun:
    torchrun --nproc_per_node=2 test_cp_conv.py
    """
    if not dist.is_initialized():
        if not setup_distributed():
            pytest.skip("Distributed environment not available. Run with torchrun.")

    try:
        run_cp_test(dtype=torch.float32)
    finally:
        pass


if __name__ == "__main__":
    if setup_distributed():
        try:
            print("Running CP Conv1d Test...")
            run_cp_test(dtype=torch.float32)
        finally:
            cleanup_distributed()
    else:
        print("Please run with torchrun:")
        print("torchrun --nproc_per_node=2 test_cp_conv.py")
