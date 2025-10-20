# Context Parallelism for Gated DeltaNet model

This directory contains the tests for Context Parallelism (CP) implementation for the Gated DeltaNet model in Flash Linear Attention.

## What is Context Parallelism?

Context Parallelism is a technique that allows splitting long sequences across multiple GPUs to handle sequences that would otherwise exceed single-GPU memory limits. Instead of splitting the model parameters (model parallelism) or batch (data parallelism), CP splits the sequence dimension.


## Files

- `fla/ops/chunk_cp.py` - Core CP operator for chunk-based computation
- `fla/layers/gated_deltanet_cp.py` - CP-enabled layer implementation
- `fla.models/modeling_gated_deltanet_cp.py` - CP-enabled model implementation  
- `tests/cp/train_cp_real.py` - Simple training with CP
- `tests/cp/benchmark_memory.py` - Memory usage comparison across different CP sizes
- `tests/cp/test_numerical_correctness.py` - Validation that CP matches single-GPU results (still working on it)
- Profiling, benchmarking speed need to be done.
- ToDo: Integrated with TorchTitan

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

## Memory Reduction Results

Here are actual benchmark results showing the memory savings with Context Parallelism:

### Single GPU (CP Size = 1)
```
Seq Len    Chunk      Model      Peak       Status    
--------------------------------------------------
1024       1024       0.14 GB    0.78 GB    ✓
2048       2048       0.16 GB    1.36 GB    ✓
4096       4096       0.16 GB    2.53 GB    ✓
8192       8192       0.16 GB    4.95 GB    ✓
16384      16384      0.16 GB    9.50 GB    ✓
```

### Context Parallelism (CP Size = 8)
```
Seq Len    Chunk      Model      Peak       Status    
--------------------------------------------------
1024       128        0.14 GB    0.31 GB    ✓  (2.5x reduction)
2048       256        0.16 GB    0.34 GB    ✓  (4.0x reduction)
4096       512        0.16 GB    0.49 GB    ✓  (5.2x reduction)
8192       1024       0.16 GB    0.78 GB    ✓  (6.3x reduction)
16384      2048       0.16 GB    1.36 GB    ✓  (7.0x reduction)
```

**Key Insights:**
- **Dramatic Memory Reduction**: Less peak memory usage with CP
- **Longer Sequences = Better Savings**: Memory reduction improves with sequence length
- **Linear Scaling**: Each GPU processes 1/8th of the sequence with 8-way CP
- **Handles Very Long Sequences**: 16K tokens easily processed with CP in much lesser time



