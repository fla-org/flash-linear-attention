# KCP: Kimi Context Parallel

Context Parallel for GDN (Gated Delta Rule) and KDA (Kimi Delta Attention).

> CP was first introduce in [PR #691](https://github.com/fla-org/flash-linear-attention/pull/691), Special thanks to [mdy666](https://github.com/mdy666)

## Core Recurrence

Both GDN and KDA share the delta rule recurrence:

```
S_t = decay(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - S_{t-1} @ k_t)
o_t = q_t^T @ S_t
```

Where `(x)` is outer product, `S` is `[K, V]` state matrix. FLA's kernels currently use `[K, V]` state matrix, but there are other backends that can store it by transposing `[V, K]`.

In the chunk-parallel formulation, we first compute the WY representation
to get `w` and `u`, then the inter-chunk state recurrence becomes:

```
h_{c+1} = decay(g_last_c) * h_c + k_c^T @ (u_c - w_c @ h_c)
```

where `c` indexes chunks, `g_last_c` is the gate value at the last token of chunk `c`.

---

## GDN vs KDA: Gate Handling

### GDN: scalar per-head gate

- `g` shape: `[B, T, H]` — one scalar per head per token
- `g` is chunk-local cumsum'd by `chunk_local_cumsum`
- All kernels receive **original** `k`, `q`, and **scalar** `g`
- Kernels internally apply gate via `USE_G=True`:
  - Decay: `h *= exp(g_last)`
  - Gate k: `k_gated = k * exp(g_last - g_token)` (done inside kernel)
  - Gate q: `q_gated = q * exp(g_token)` (done inside kernel, backward only)

### KDA: per-dim gate

- `g` shape: `[B, T, H, K]` — one scalar per dimension per token
- `g` is chunk-local cumsum'd + scaled by `kda_gate_chunk_cumsum` (includes gate activation)
  or `chunk_local_cumsum` (if gate pre-computed)
- The WY repr step (`chunk_kda_fwd_intra` / `recompute_w_u_fwd`) pre-computes gated tensors:
  - `kg = k * exp2(gk_last_chunk - gk)` — relative gate, k aligned to chunk end
  - `qg = q * exp2(gk)` — absolute gate on q (used in backward only)
- All kernels receive **pre-gated** `kg` (and `qg` in backward), plus `gk=g` for inter-chunk decay
- Kernels apply only the **chunk-level** decay via `USE_GK=True`:
  - Decay: `h *= exp2(gk_last)` (diagonal, per-dim)
  - No further gating on k/q — already done externally

**Why the difference**: GDN's scalar gate is cheap to apply inside kernels. KDA's per-dim
gate `[H, K]` is more efficiently pre-applied during the WY representation step.

---

## CP Architecture

### Data Flow

Each rank holds a local chunk of the sequence. CP computes cross-rank initial states
via an all-gather + merge pattern:

```
1. Each rank computes local (h_ext, M) from its chunk
   - h_ext: accumulated state assuming h_0 = 0
   - M: transition matrix (product of per-chunk transition)

2. All-gather [h_ext, M] across all ranks

3. Rank r merges from ranks < r:
   h_r = fold over [rank_{r-pre}, ..., rank_{r-1}]:
       h = M_j @ h + h_ext_j
```

### Pre-Process Forward

Computes `(h_ext, M)` for the first sequence of the local chunk.

**Stage 1 — h_ext `[K, V]`:**

```
h = 0
for each sub-chunk c:
    h *= decay(g_last_c)    # inter-chunk decay
    v_new = u_c - w_c @ h   # (computed via w @ h subtraction)
    h += k_c^T @ v_new      # accumulate
```

**Stage 2 — M `[K, K]` (transition matrix):**

```
M = I
for each sub-chunk c:
    M_c = diag(decay(g_last_c)) - k_c^T @ w_c
    M = M_c @ M              # chain multiply
```

**Merge (forward direction):**

For rank r with `pre_num_ranks` previous ranks:
```
h = 0
for j from (r - pre_num_ranks) to (r - 1):
    h = M_j @ h + h_ext_j
```

### Pre-Process Backward

Same structure but **reversed** direction — merges from ranks **after** current rank.

**Stage 1 — dh_ext `[K, V]`:**

```
dh = 0
for each sub-chunk c (reverse order):
    dh *= decay(g_last_c)
    dv = k_c @ dh + original_dv_c
    dh += q_c^T @ do_c * scale - w_c^T @ dv
```

**Stage 2 — dM `[K, K]`:**

```
dM = I
for each sub-chunk c (reverse order):
    dM_c = diag(decay(g_last_c)) - w_c^T @ k_c     # NOTE: transposed vs forward
    dM = dM_c @ dM
```

**Merge (backward direction):**

For rank r with `post_num_ranks` following ranks:
```
dh = 0
for j from (r + post_num_ranks) down to (r + 1):
    dh = dM_j @ dh + dh_ext_j
```

---

## Actual Code Flow

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
# kg = k * exp2(gk_last_chunk - gk)   [relative gate to chunk end]
# qg = q * exp2(gk)                   [absolute gate, saved for backward]

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
# qg = q * exp2(gk)
# kg = k * exp2(gk_last_chunk - gk)

# 2. Recompute h
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

| Function | GDN | KDA | Gate Path |
|----------|-----|-----|-----------|
| pre_process_fwd | k=k, g=g | k=kg, gk=g | GDN: USE_G, KDA: USE_GK |
| fwd_h | k=k, g=g | k=kg, gk=g | Same as pre_process |
| pre_process_bwd | q=q, k=k, g=g | q=qg, k=kg, gk=g | GDN: USE_G, KDA: USE_GK |
| bwd_dhu | q=q, k=k, g=g | q=qg, k=kg, gk=g | Same as pre_process |

**Key consistency**: pre_process and main kernel always receive the **same** tensors.
For KDA, both receive pre-gated `kg`/`qg`. For GDN, both receive original tensors.

---

## M (Transition Matrix)

The transition matrix captures how the state transforms across a chunk:

```
M_c = diag(decay) - k_c^T @ w_c     (forward)
dM_c = diag(decay) - w_c^T @ k_c    (backward, transposed)
```

Where `decay` is:
- GDN: `exp(g_last)` scalar → `diag(exp(g_last)) = exp(g_last) * I`
- KDA: `exp2(gk_last)` per-dim → `diag(exp2(gk_last_0), exp2(gk_last_1), ...)`

Cross-rank state is computed by chaining M matrices:
```
h_r = M_{r-1} @ (M_{r-2} @ (... @ h_ext_0 + h_ext_1) + ...) + h_ext_{r-1}
```

**Precision note**: The M chain multiply `b_m = M_i @ b_m` must stay in fp32 to
avoid accumulated precision loss. In bf16, repeatedly casting fp32 accumulators back
to bf16 between iterations causes significant error growth over many chunks.

---

## compress_h0 / expand_h0

Optimization for CP mode. Since only the first sequence in the local batch can be
a continuation from a previous rank, only its initial_state is non-zero.
`compress_h0` extracts just that one state to save memory during `save_for_backward`.
`expand_h0` restores the full `[N, H, K, V]` tensor in backward.
