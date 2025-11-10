# Context Parallelism for Gated DeltaNet model

This directory contains the tests for Context Parallelism (CP) implementation for the Gated DeltaNet model in Flash Linear Attention.

## What is Context Parallelism?

Context Parallelism is a technique that allows splitting long sequences across multiple GPUs to handle sequences that would otherwise exceed single-GPU memory limits. Instead of splitting the model parameters (model parallelism) or batch (data parallelism), CP splits the sequence dimension.

## Files

- `fla/ops/chunk_cp.py` - Core CP operator for chunk-based computation
- `fla/layers/gated_deltanet_cp.py` - CP-enabled layer implementation
- `fla/models/modeling_gated_deltanet_cp.py` - CP-enabled model implementation  
- `tests/cp/train_cp_real.py` - Simple training with CP
- `tests/cp/benchmark_memory.py` - Memory usage comparison across different CP sizes
- `tests/cp/test_numerical_correctness.py` - Validation that CP matches single-GPU results 
- Profiling, benchmarking speed need to be done.
- ToDo: Integrate with TorchTitan

## Quick Start

### 1. Train with CP

```


torchrun --nproc_per_node=8 tests/cp/train_cp_real.py --cp_size 8

torchrun --nproc_per_node=4 tests/cp/train_cp_real.py --cp_size 4 --seq_len=4096 --batch_size=1


```


### 2. Benchmark Memory Usage


```


# Single GPU baseline
python tests/cp/benchmark_memory.py --cp_size 1

# Compare with CP
torchrun --nproc_per_node=4 tests/cp/benchmark_memory.py --cp_size 4
```

**Key Insights:**
- **Dramatic Memory Reduction**: Less peak memory usage with CP
- **Longer Sequences = Better Savings**: Memory reduction improves with sequence length
- **Linear Scaling**: Each GPU processes 1/8th of the sequence with 8-way CP
- **Handles Very Long Sequences**: 16K tokens easily processed with CP in much lesser time



### 3. Validation


```


# Step 1: Generate reference on single GPU
python test_numerical_correctness.py --mode single --save_ref

# Step 2: Test with CP
torchrun --nproc_per_node=2 test_numerical_correctness.py --mode cp --cp_size 2
```