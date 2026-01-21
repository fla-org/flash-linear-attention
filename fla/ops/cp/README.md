# CP Related Features was first implemented by Duyue MA
# And integrated by Zhiyuan Li

# Context Parallel (CP) Usage Guide

## Conventions
- CP splits the sequence dimension across ranks. Each rank owns a local chunk
  of tokens and runs the operator on that local chunk.
- CP context stores **rank-local** varlen metadata:
  - `FLACPContext.cu_seqlens` is rank-local, on GPU (int64 / torch.long).
  - `FLACPContext.cu_seqlens_cpu` is rank-local, on CPU (int64 / torch.long).
- Variable-length inputs are represented by global `cu_seqlens` **before**
  partition; `build_cp_context` converts them into rank-local metadata.
- CP runs do **not** support `initial_state` or `output_final_state=True`.

## Build CP Context
```python
from fla.ops.cp import build_cp_context

# global cu_seqlens before partition (device can be CPU or GPU)
cu_seqlens_global = torch.tensor([0, s1, s1+s2, ..., total], dtype=torch.long, device=device)

# conv1d_kernel_size is required for causal_conv1d CP path
cp_context = build_cp_context(
    cu_seqlens_global,
    group=dist.group.WORLD,
    conv1d_kernel_size=W,
)
```

## Causal Conv1d (CP)
```python
from fla.modules.convolution import causal_conv1d

# x_local is the rank-local chunk: [1, T_local, D]
y_local, _ = causal_conv1d(
    x=x_local,
    weight=weight_local,
    bias=bias_local,
    activation="swish",
    cp_context=cp_context,
)
```
Notes:
- `cp_context` is required.
- `cp_context.conv1d_kernel_size` and `cp_context.cu_seqlens` must be set.
- Do not pass `cu_seqlens`/`cu_seqlens_cpu` manually; they are taken from context.

## KDA (CP)
```python
from fla.ops.kda import chunk_kda

o_local, _ = chunk_kda(
    q=F.normalize(q_local, p=2, dim=-1),
    k=F.normalize(k_local, p=2, dim=-1),
    v=v_local,
    g=g_local,
    beta=beta_local,
    cp_context=cp_context,
    disable_recompute=disable_recompute,
)
```
Notes:
- CP expects `B == 1` for varlen and uses rank-local `cu_seqlens` from context.
- `initial_state` and `output_final_state=True` are not supported in CP.

## Test References
- `tests/context_parallel/test_cp_conv.py`
- `tests/context_parallel/test_cp_kda.py`
