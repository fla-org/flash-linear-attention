# Context Parallel of Linear Atttention

Context Parallel of Linear Atttention (alias KCP in Moonshot) is context parallelism designed for delta-rule recurrent models such as GDN (Gated Delta Rule) and KDA (Kimi Delta Attention). It enables efficient distributed training by partitioning the sequence dimension across ranks, with each rank processing a local token chunk and CP automatically synchronizing cross-rank states.

## Quick Start

### Build CP Context

```python
from fla.ops.cp import build_cp_context

# global cu_seqlens before partition (device can be CPU or GPU)
cu_seqlens_global = torch.tensor(
    [0, s1, s1 + s2, ..., total],
    dtype=torch.long,
    device=device
)

# conv1d_kernel_size is required for causal_conv1d CP path
cp_context = build_cp_context(
    cu_seqlens_global,
    group=dist.group.WORLD,
    conv1d_kernel_size=W,
)
```

### Causal Conv1d

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

> [!NOTE]
> - `cp_context` is required; `cp_context.conv1d_kernel_size` and `cp_context.cu_seqlens` must be set.
> - Do not pass `cu_seqlens` / `cu_seqlens_cpu` manually — they are taken from context.

### KDA

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

> [!NOTE]
> - CP expects `B == 1` for varlen and uses rank-local `cu_seqlens` from context.
> - `initial_state` and `output_final_state=True` are not supported in CP mode.

---

## Conventions

CP context stores **rank-local** varlen metadata that tracks how sequences are distributed:

- `FLACPContext.cu_seqlens` — rank-local cumulative sequence lengths, on GPU (`int64`)
- `FLACPContext.cu_seqlens_cpu` — same data on CPU for host-side indexing

Variable-length inputs start as global `cu_seqlens` **before** partitioning; `build_cp_context` converts them into rank-local metadata automatically.

---

## Notation

We follow the notation from the [Kimi Linear technical report](https://yzhang.site/assets/pubs/techreport/2025/kda.pdf) (Section 2.1). Throughout this document, subscript $[t]$ denotes chunk index, while subscript $t$ denotes token position.

**Vectors and matrices:**
- $\boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t, \boldsymbol{o}_t, \boldsymbol{u}_t, \boldsymbol{w}_t$ — column vectors in $\mathbb{R}^{d_k}$ or $\mathbb{R}^{d_v}$ at position $t$
- $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$ — matrix-form memory state; FLA kernels store as `[d_k, d_v]`, some backends transpose to `[d_v, d_k]`
- $\mathbf{X}$ with subscript $[t]$ — stacked vectors within chunk $t$ (shape $C \times d$); sequence length $L$ splits into $L/C$ chunks of size $C$
- $\boldsymbol{x}^r$ with subscript $[t]$ — the $r$-th element in chunk $t$, i.e., $\boldsymbol{x}_{tC+r}$ where $t \in [0, L/C), r \in [1, C]$

**State and decay:**

$$\mathbf{S}_{[t]} := \mathbf{S}_{[t]}^{0} = \mathbf{S}_{[t-1]}^{C} \quad \text{(state at chunk boundary)}$$

$$\gamma^{i \to j}_{[t]} := \prod_{k=i}^j \alpha^k_{[t]} \quad \text{(cumulative decay from position } i \text{ to } j \text{)}$$

$$\gamma^r_{[t]} := \gamma^{1 \to r}_{[t]} \quad \text{(shorthand)}$$

$$\boldsymbol{\Gamma}^{i \to j}_{[t]} \in \mathbb{R}^{C \times d_k} \quad \text{(stacked rows from } \gamma^i_{[t]} \text{ to } \gamma^j_{[t]} \text{)}$$

The decay factor $\alpha_t$ is a scalar $\in [0,1]$ for GDN, or per-dim $\in [0,1]^{d_k}$ for KDA.

**Code mapping:**
- `g` stores $\log(\alpha)$ (or $\log_2(\alpha)$ for KDA)
- After `chunk_local_cumsum`, `g` at position $r$ equals $\log \gamma^r$
- Then $\exp(g) = \gamma$ and $\exp(g_{\mathrm{last}} - g_r) = \gamma^{r \to C}$

---

## Recurrence

Both GDN and KDA are built on the **delta rule** — a recurrent update where the state matrix $\mathbf{S}$ is first decayed, then updated by subtracting the old key's contribution and adding the new one. This enables efficient "memory editing" where stale information can be forgotten.

### GDN — Scalar Per-Head Gate

GDN uses a single scalar gate per head per token. From [Yang et al., 2025]:

$$\mathbf{S}_t = \alpha_t (\mathbf{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \mathbf{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top$$

$$\boldsymbol{o}_t = \mathbf{S}_t^\top \boldsymbol{q}_t$$

### KDA — Per-Dim Gate

KDA extends GDN with a per-dimension gate, giving finer control over which features to retain or forget. From Eq. 1 in the Kimi-k1.5 report:

$$\mathbf{S}_t = (\mathbf{I} - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) \mathrm{Diag}(\alpha_t) \mathbf{S}_{t-1} + \beta_t \boldsymbol{k}_t \boldsymbol{v}_t^\top$$

$$\boldsymbol{o}_t = \mathbf{S}_t^\top \boldsymbol{q}_t$$

### Chunkwise Formulation

For efficiency, we process tokens in chunks using the WY representation (Eq. 7 in the report), which computes auxiliary matrices $\mathbf{W}, \mathbf{U}$ for each chunk. The inter-chunk state recurrence (Eq. 8) becomes:

$$\mathbf{S}_{[t+1]} = \mathrm{Diag}(\gamma^C_{[t]}) \mathbf{S}_{[t]} + (\boldsymbol{\Gamma}^{i \to C}_{[t]} \odot \mathbf{K}_{[t]})^\top (\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]})$$

This formulation is key to CP: it lets us compute how the state transforms across a chunk, enabling efficient cross-rank synchronization.

---

## GDN vs KDA: Gate Handling

While both models share the delta rule structure, they differ in how gating is applied — a distinction that affects the CP implementation.

### GDN

GDN's scalar gate is cheap to apply inside kernels, so we pass the **original** tensors and let the kernel handle gating internally:

- **Gate**: $\alpha_t \in [0,1]$, one scalar per head per token
- **Code**: `g` shape `[B, T, H]` where $\alpha = \exp(g)$; processed by `chunk_local_cumsum`
- **Kernel input**: Original $\boldsymbol{k}$, $\boldsymbol{q}$, and scalar `g`
- **Internal gating** (`USE_G=True`):
  - Inter-chunk decay: $\mathbf{S} \leftarrow \gamma^C \cdot \mathbf{S}$ (scalar broadcast)
  - Gated key: $\tilde{\boldsymbol{k}}^r = \boldsymbol{k}^r \cdot \gamma^{r \to C}$
  - Gated query: $\tilde{\boldsymbol{q}}^r = \boldsymbol{q}^r \cdot \gamma^r$ (backward only)

### KDA

KDA's per-dim gate $\mathrm{Diag}(\alpha_t) \in \mathbb{R}^{d_k \times d_k}$ would be expensive to apply inside kernels. Instead, we **pre-gate** the tensors during the WY representation step:

- **Gate**: $\alpha_t \in [0,1]^{d_k}$, one value per dimension per token
- **Code**: `g` shape `[B, T, H, K]` where $\alpha = \exp_2(g)$; processed by `kda_gate_chunk_cumsum`
- **Pre-gated tensors** (from `chunk_kda_fwd_intra` / `recompute_w_u_fwd`):
  - `kg`: row $r$ is $\boldsymbol{k}^r \odot \gamma^{r \to C}$, i.e., `k * exp2(gk_last - gk)`
  - `qg`: row $r$ is $\boldsymbol{q}^r \odot \gamma^{r}$, i.e., `q * exp2(gk)` (saved for backward)
- **Kernel input**: Pre-gated `kg` (and `qg` in backward), plus `gk=g` for inter-chunk decay
- **Kernel gating** (`USE_GK=True`): Only chunk-level decay $\mathbf{S} \leftarrow \mathrm{Diag}(\gamma^C) \mathbf{S}$

This design means CP pre-processing must use the **same** tensors as the main kernel — original for GDN, pre-gated for KDA.

---

## CP Architecture

### Data Flow

The core challenge of CP is that each rank only sees a local chunk, but the recurrent state depends on all previous tokens. We solve this with an **all-gather + merge** pattern:

1. **Local computation**: Each rank computes $(\mathbf{S}_\text{ext}, \mathbf{M})$ from its chunk
   - $\mathbf{S}_\text{ext} \in \mathbb{R}^{d_k \times d_v}$: accumulated state assuming $\mathbf{S}_0 = \mathbf{0}$
   - $\mathbf{M} \in \mathbb{R}^{d_k \times d_k}$: transition matrix capturing how the chunk transforms incoming state

2. **All-gather**: Collect $[\mathbf{S}_\text{ext}, \mathbf{M}]$ from all ranks

3. **Merge**: Rank $r$ reconstructs its initial state by chaining contributions from ranks $< r$:

$$\mathbf{S} = \mathbf{0}; \quad \text{for } j = (r - n_\text{pre}) \text{ to } (r-1): \quad \mathbf{S} \leftarrow \mathbf{M}_j \mathbf{S} + \mathbf{S}_{\text{ext},j}$$

### Pre-Process Forward

This step computes $(\mathbf{S}_\text{ext}, \mathbf{M})$ for the local chunk.

**Stage 1 — Accumulated state $\mathbf{S}_\text{ext} \in \mathbb{R}^{d_k \times d_v}$:**

We simulate processing the chunk with zero initial state. Initialize $\mathbf{S} = \mathbf{0}$, then for each sub-chunk $(t)$:

$$\mathbf{S} \leftarrow \mathrm{Diag}(\gamma^C_{[t]}) \mathbf{S} + (\boldsymbol{\Gamma}^{i \to C}_{[t]} \odot \mathbf{K}_{[t]})^\top (\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S})$$

**Stage 2 — Transition matrix $\mathbf{M} \in \mathbb{R}^{d_k \times d_k}$:**

The transition matrix captures how incoming state is transformed. Initialize $\mathbf{M} = \mathbf{I}$, then for each sub-chunk $(t)$:

$$\mathbf{M}_{[t]} = \mathrm{Diag}(\gamma^C_{[t]}) - (\boldsymbol{\Gamma}^{i \to C}_{[t]} \odot \mathbf{K}_{[t]})^\top \mathbf{W}_{[t]}$$

$$\mathbf{M} \leftarrow \mathbf{M}_{[t]} \mathbf{M}$$

**Merge (forward direction):**

For rank $r$ with `pre_num_ranks` previous ranks:

$$\mathbf{S} = \mathbf{0}; \quad \text{for } j = (r - n_\text{pre}) \text{ to } (r-1): \quad \mathbf{S} \leftarrow \mathbf{M}_j \mathbf{S} + \mathbf{S}_{\text{ext},j}$$

### Pre-Process Backward

The backward pass has the same structure but **reversed** direction — we merge from ranks **after** the current rank to propagate gradients backward through the sequence.

**Stage 1 — Gradient $\mathrm{d}\mathbf{S}_\text{ext} \in \mathbb{R}^{d_k \times d_v}$:**

Initialize $\mathrm{d}\mathbf{S} = \mathbf{0}$. For each sub-chunk $(t)$ in reverse order:

$$\mathrm{d}\mathbf{S} \leftarrow \mathrm{Diag}(\gamma^C_{[t]}) \mathrm{d}\mathbf{S}$$

$$\mathrm{d}\mathbf{V} = \mathbf{K}_{[t]} \mathrm{d}\mathbf{S} + \mathrm{d}\mathbf{V}_\text{local}$$

$$\mathrm{d}\mathbf{S} \leftarrow \mathrm{d}\mathbf{S} + (\boldsymbol{\Gamma}^{1 \to C}_{[t]} \odot \mathbf{Q}_{[t]})^\top \mathrm{d}\mathbf{O}_{[t]} \cdot s - \mathbf{W}_{[t]}^\top \mathrm{d}\mathbf{V}$$

where $s = d_k^{-1/2}$ is the scaling factor.

**Stage 2 — Gradient $\mathrm{d}\mathbf{M} \in \mathbb{R}^{d_k \times d_k}$:**

Initialize $\mathrm{d}\mathbf{M} = \mathbf{I}$. For each sub-chunk $(t)$ in reverse order:

$$\mathrm{d}\mathbf{M}_{[t]} = \mathrm{Diag}(\gamma^C_{[t]}) - \mathbf{W}_{[t]}^\top (\boldsymbol{\Gamma}^{i \to C}_{[t]} \odot \mathbf{K}_{[t]})$$

$$\mathrm{d}\mathbf{M} \leftarrow \mathrm{d}\mathbf{M}_{[t]} \mathrm{d}\mathbf{M}$$

> [!NOTE]
> $\mathrm{d}\mathbf{M}$ is the transpose of forward $\mathbf{M}$ ($\mathbf{W}^\top \mathbf{K}$ vs. $\mathbf{K}^\top \mathbf{W}$).

**Merge (backward direction):**

For rank $r$ with `post_num_ranks` following ranks:

$$\mathrm{d}\mathbf{S} = \mathbf{0}; \quad \text{for } j = (r + n_\text{post}) \text{ down to } (r+1): \quad \mathrm{d}\mathbf{S} \leftarrow \mathrm{d}\mathbf{M}_j \mathrm{d}\mathbf{S} + \mathrm{d}\mathbf{S}_{\text{ext},j}$$

---

## Code Flow

The following examples show how CP integrates with the existing kernel interfaces.

### GDN Forward

```python
g = chunk_local_cumsum(g, chunk_size=64)
w, u = recompute_w_u_fwd(k, v, beta, A, g=g)

# CP pre-process: original k, scalar g
initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
    k=k, w=w, u=u, g=g,       # USE_G=True, USE_GK=False
    context=cp_context,
)

# Main kernel: original k, scalar g
h, v_new, _ = chunk_gated_delta_rule_fwd_h(
    k=k, w=w, u=u, g=g,
    initial_state=initial_state,
)
```

### GDN Backward

```python
w, u = recompute_w_u_fwd(k, v, beta, A, g=g)
h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, ...)
dv = chunk_bwd_dv_local(q=q, k=k, g=g, do=do, ...)

# CP pre-process: original q, k, scalar g
dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
    q=q, k=k, w=w, do=do, dv=dv, g=g,    # USE_G=True, USE_GK=False
    context=cp_context,
)

# Main kernel: original q, k, scalar g
dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
    q=q, k=k, w=w, g=g,
    dht=dht, ...
)
```

### KDA Forward

```python
# 1. Intra-chunk: compute WY repr + pre-gated tensors
w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(q, k, v, gk=g, beta, ...)
# kg = K ⊙ exp2(γ^{r→C}), i.e., rows of Γ^{i→C} ⊙ K
# qg = Q ⊙ exp2(γ^r),     i.e., rows of Γ^{1→C} ⊙ Q (saved for backward)

# 2. CP pre-process: pre-gated kg, per-dim gk=g
initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
    k=kg, w=w, u=u, gk=g,     # USE_G=False, USE_GK=True, use_exp2=True
    context=cp_context,
)

# 3. Main kernel: pre-gated kg, per-dim gk=g
h, v_new, _ = chunk_gated_delta_rule_fwd_h(
    k=kg, w=w, u=u, gk=g,
    initial_state=initial_state,
    use_exp2=True,
)
```

### KDA Backward

```python
# 1. Recompute WY repr
w, u, qg, kg = recompute_w_u_fwd(q, k, v, beta, A=Akk, gk=g, ...)
# qg = Q ⊙ exp2(γ^r), kg = K ⊙ exp2(γ^{r→C})

# 2. Recompute state
h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g, ...)

# 3. Compute local dv
dAqk, dv = chunk_kda_bwd_dAv(q, k, v=v_new, do, A=Aqk, ...)

# 4. CP pre-process: pre-gated qg, kg, per-dim gk=g
dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
    q=qg, k=kg, w=w, do=do, dv=dv, gk=g,  # USE_G=False, USE_GK=True, use_exp2=True
    context=cp_context,
)

# 5. Main kernel: pre-gated qg, kg
dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
    q=qg, k=kg, w=w, gk=g,
    dht=dht, ...
    use_exp2=True,
)
```

---

## Input Tensor Summary

| Function          | GDN                 | KDA                    | Gate Path                   |
| ----------------- | ------------------- | ---------------------- | --------------------------- |
| `pre_process_fwd` | `k=k`, `g=g`        | `k=kg`, `gk=g`         | GDN: `USE_G`, KDA: `USE_GK` |
| `fwd_h`           | `k=k`, `g=g`        | `k=kg`, `gk=g`         | Same as pre_process         |
| `pre_process_bwd` | `q=q`, `k=k`, `g=g` | `q=qg`, `k=kg`, `gk=g` | GDN: `USE_G`, KDA: `USE_GK` |
| `bwd_dhu`         | `q=q`, `k=k`, `g=g` | `q=qg`, `k=kg`, `gk=g` | Same as pre_process         |

**Key consistency:** Pre-process and main kernel must always receive the **same** tensors — this is critical for correctness.
- **KDA**: Both receive pre-gated `kg` ($\boldsymbol{\Gamma}^{i \to C} \odot \mathbf{K}$) and `qg` ($\boldsymbol{\Gamma}^{1 \to C} \odot \mathbf{Q}$)
- **GDN**: Both receive original $\boldsymbol{k}$, $\boldsymbol{q}$ (gating applied inside the kernel)

---

## Transition Matrix

The transition matrix $\mathbf{M}$ is central to CP — it captures how a chunk transforms any incoming state, enabling us to chain contributions from multiple ranks.

**Forward:**

$$\mathbf{M}_{[t]} = \mathrm{Diag}(\gamma^C_{[t]}) - (\boldsymbol{\Gamma}^{i \to C}_{[t]} \odot \mathbf{K}_{[t]})^\top \mathbf{W}_{[t]}$$

**Backward (transposed):**

$$\mathrm{d}\mathbf{M}_{[t]} = \mathrm{Diag}(\gamma^C_{[t]}) - \mathbf{W}_{[t]}^\top (\boldsymbol{\Gamma}^{i \to C}_{[t]} \odot \mathbf{K}_{[t]})$$

The diagonal term $\mathrm{Diag}(\gamma^C)$ differs between models:
- **GDN**: $\exp(g_{\mathrm{last}}) \cdot \mathbf{I}$ — scalar times identity
- **KDA**: $\mathrm{Diag}(\gamma^C)$ — per-dim diagonal, where $\gamma^C = \exp_2(g_{\mathrm{last}})$ (i.e., `gk_last` in code)

Cross-rank state is computed by chaining $\mathbf{M}$ matrices:

$$\mathbf{S}_r = \mathbf{M}_{r-1} (\mathbf{M}_{r-2} (\cdots \mathbf{S}_{\text{ext},0} + \mathbf{S}_{\text{ext},1}) + \cdots) + \mathbf{S}_{\text{ext},r-1}$$

> [!IMPORTANT]
> The $\mathbf{M}$ chain multiply must stay in **fp32** to avoid accumulated precision loss. In bf16, repeatedly casting fp32 accumulators back to bf16 between iterations causes significant error growth over many chunks.

---

## Initial State Memory Optimization

In CP mode, only the first sequence in the local batch can be a continuation from a previous rank — all other sequences start fresh. This means only one initial state $\mathbf{S}_0 \in \mathbb{R}^{H \times d_k \times d_v}$ is non-zero, presenting an opportunity for memory savings:

- `compress_h0`: Extracts just that one state to save memory during `save_for_backward`
- `expand_h0`: Restores the full `[N, H, d_k, d_v]` tensor in backward

---

## Test References

- [`tests/context_parallel/test_cp_conv.py`](../../../tests/context_parallel/test_cp_conv.py)
- [`tests/context_parallel/test_cp_kda.py`](../../../tests/context_parallel/test_cp_kda.py)

## Discussion

While this document focuses on delta-rule models such as GDN and KDA, the underlying CP mechanism is **not restricted to delta-rule recurrences**. In fact, any linear attention formulation that can be expressed in a *chunkwise* form — i.e., one where the state transition across a chunk can be decomposed into a transition matrix $\mathbf{M}$ and an accumulated state $\mathbf{S}_\text{ext}$ — can adopt the same **pre-process + all-gather + merge** strategy for context parallelism.

The only model-specific components are:
1. How $\mathbf{M}$ and $\mathbf{S}_\text{ext}$ are computed from the local chunk.
2. How the merge kernel chains these quantities across ranks.

As long as these two operations are well-defined, the same CP infrastructure (`build_cp_context`, all-gather, and merge) applies without changing the high-level data flow.

At the time of writing, CP has been implemented and verified for **GDN**, **KDA**, and **DPLR** (a.k.a. RWKV-7). If you would like to see support for another linear-attention variant, please feel free to open an issue.

## Acknowledgments

Context Parallel of Linear Attention was first introduced in [PR #691](https://github.com/fla-org/flash-linear-attention/pull/691), implemented by [Duyue MA](https://github.com/mdy666). It is also known as **KCP** (Kimi Context Parallel) internally at Moonshot AI. The implementation in this repository was independently contributed to FLA and is a separate codebase from the internal Moonshot implementation.
