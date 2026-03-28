# KCP: Kimi Context Parallel

Context Parallel for GDN (Gated Delta Rule) and KDA (Kimi Delta Attention).

> CP was first introduced in [PR #691](https://github.com/fla-org/flash-linear-attention/pull/691). Special thanks to [mdy666](https://github.com/mdy666).

## Notation

Following the [Kimi Linear technical report](https://yzhang.site/assets/pubs/techreport/2025/kda.pdf) (Section 2.1):

- **Vectors**: $\square_t \in \mathbb{R}^{d_k}$ or $\mathbb{R}^{d_v}$ for $\square \in \{\mathbf{q}, \mathbf{k}, \mathbf{v}, \mathbf{o}, \mathbf{u}, \mathbf{w}\}$ denotes the $t$-th column vector.
- **State**: $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$ is the matrix-form memory state. FLA kernels store it as $[d_k, d_v]$; some other backends transpose to $[d_v, d_k]$.
- **Chunk indexing**: The sequence of length $L$ is split into $L/C$ chunks of size $C$. $\square_{[t]} \in \mathbb{R}^{C \times d}$ stacks vectors within chunk $t$; $\square^r_{[t]} = \square_{tC+r}$ is the $r$-th element ($t \in [0, L/C)$, $r \in [1, C]$). State: $\mathbf{S}_{[t]} := \mathbf{S}^0_{[t]} = \mathbf{S}^C_{[t-1]}$.
- **Decay**: $\alpha_t \in [0,1]$ (GDN, scalar) or $\alpha_t \in [0,1]^{d_k}$ (KDA, per-dim). Cumulative decay $\gamma^{i \to j}_{[t]} := \prod_{k=i}^j \alpha^k_{[t]}$, abbreviated $\gamma^r_{[t]} := \gamma^{1 \to r}_{[t]}$. $\text{Diag}(\gamma^{i \to j}_{[t]}) := \prod_{k=i}^j \text{Diag}(\alpha^k_{[t]})$. $\Gamma^{i \to j}_{[t]} \in \mathbb{R}^{C \times d_k}$ stacks rows from $\gamma^i_{[t]}$ to $\gamma^j_{[t]}$.
- **Code mapping**: In code, `g` stores $\log(\alpha)$ (or $\log_2(\alpha)$ for KDA). After `chunk_local_cumsum`, `g` at position $r$ equals $\log \gamma^r_{[t]}$. Then $\exp(\texttt{g}) = \gamma$ and $\exp(\texttt{g\_last} - \texttt{g}_r) = \gamma^{r \to C}_{[t]}$.

---

## Recurrence

**GDN** — scalar per-head gate (Eq. from [Yang et al., 2025]):

$$\mathbf{S}_t = \alpha_t (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top, \quad \mathbf{o}_t = \mathbf{S}_t^\top \mathbf{q}_t$$

**KDA** — per-dim gate (Eq. 1 in the report):

$$\mathbf{S}_t = (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \, \text{Diag}(\alpha_t) \, \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top, \quad \mathbf{o}_t = \mathbf{S}_t^\top \mathbf{q}_t$$

In the chunkwise formulation, the WY representation (Eq. 7) computes auxiliary matrices $\mathbf{W}_{[t]}$ and $\mathbf{U}_{[t]}$. The inter-chunk state recurrence (Eq. 8) is:

$$\mathbf{S}_{[t+1]} = \text{Diag}(\gamma^C_{[t]}) \, \mathbf{S}_{[t]} + \left(\Gamma^{i \to C}_{[t]} \odot \mathbf{K}_{[t]}\right)^\top \left(\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \, \mathbf{S}_{[t]}\right)$$

---

## GDN vs KDA: Gate Handling

### GDN: scalar per-head gate

- $\alpha_t \in [0,1]$, one scalar per head per token
- In code: `g` shape `[B, T, H]` where $\alpha = \exp(g)$; processed by `chunk_local_cumsum`
- All kernels receive **original** $\mathbf{k}$, $\mathbf{q}$, and **scalar** `g`
- Kernels internally apply gating via `USE_G=True`:
  - Inter-chunk decay: $\mathbf{S} \leftarrow \gamma^C_{[t]} \cdot \mathbf{S}$ (scalar broadcast)
  - Gated key: $\tilde{\mathbf{k}}^r_{[t]} = \mathbf{k}^r_{[t]} \cdot \gamma^{r \to C}_{[t]}$ (done inside kernel)
  - Gated query: $\tilde{\mathbf{q}}^r_{[t]} = \mathbf{q}^r_{[t]} \cdot \gamma^r_{[t]}$ (done inside kernel, backward only)

### KDA: per-dim gate

- $\alpha_t \in [0,1]^{d_k}$, one value per dimension per token
- In code: `g` shape `[B, T, H, K]` where $\alpha = \exp_2(g)$; processed by `kda_gate_chunk_cumsum` (includes gate activation) or `chunk_local_cumsum` (if pre-computed)
- The WY representation step (`chunk_kda_fwd_intra` / `recompute_w_u_fwd`) pre-computes gated tensors:
  - `kg`: row $r$ is $\mathbf{k}^r_{[t]} \odot \gamma^{r \to C}_{[t]}$, i.e., `k * exp2(gk_last - gk)`. In matrix form: $\Gamma^{i \to C}_{[t]} \odot \mathbf{K}_{[t]}$.
  - `qg`: row $r$ is $\mathbf{q}^r_{[t]} \odot \gamma^{r}_{[t]}$, i.e., `q * exp2(gk)`. In matrix form: $\Gamma^{1 \to C}_{[t]} \odot \mathbf{Q}_{[t]}$ (saved for backward).
- All kernels receive **pre-gated** `kg` (and `qg` in backward), plus `gk=g` for inter-chunk decay
- Kernels apply only the **chunk-level** decay via `USE_GK=True`:
  - Inter-chunk decay: $\mathbf{S} \leftarrow \text{Diag}(\gamma^C_{[t]}) \, \mathbf{S}$ (per-dim diagonal)
  - No further gating on $\mathbf{k}$/$\mathbf{q}$ — already done externally

**Why the difference**: GDN's scalar gate is cheap to apply inside kernels. KDA's per-dim gate $\text{Diag}(\alpha_t) \in \mathbb{R}^{d_k \times d_k}$ is more efficiently pre-applied during the WY representation step.

---

## CP Architecture

### Data Flow

Each rank holds a local chunk of the sequence. CP computes cross-rank initial states via an all-gather + merge pattern:

1. Each rank computes local $(\mathbf{S}_\text{ext}, \mathbf{M})$ from its chunk
   - $\mathbf{S}_\text{ext} \in \mathbb{R}^{d_k \times d_v}$: accumulated state assuming $\mathbf{S}_0 = \mathbf{0}$
   - $\mathbf{M} \in \mathbb{R}^{d_k \times d_k}$: transition matrix (product of per-chunk transitions)

2. All-gather $[\mathbf{S}_\text{ext}, \mathbf{M}]$ across all ranks

3. Rank $r$ merges from ranks $< r$:

$$\mathbf{S} = \mathbf{0}; \quad \text{for } j \text{ from } (r - n_\text{pre}) \text{ to } (r-1): \quad \mathbf{S} \leftarrow \mathbf{M}_j \, \mathbf{S} + \mathbf{S}_{\text{ext},j}$$

### Pre-Process Forward

Computes $(\mathbf{S}_\text{ext}, \mathbf{M})$ for the local chunk.

**Stage 1 — $\mathbf{S}_\text{ext} \in \mathbb{R}^{d_k \times d_v}$** (accumulated state):

$$\mathbf{S} = \mathbf{0}$$

For each sub-chunk $[t]$:

$$\mathbf{S} \leftarrow \text{Diag}(\gamma^C_{[t]}) \, \mathbf{S} + \left(\Gamma^{i \to C}_{[t]} \odot \mathbf{K}_{[t]}\right)^\top \left(\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \, \mathbf{S}\right)$$

**Stage 2 — $\mathbf{M} \in \mathbb{R}^{d_k \times d_k}$** (transition matrix):

$$\mathbf{M} = \mathbf{I}$$

For each sub-chunk $[t]$:

$$\mathbf{M}_{[t]} = \text{Diag}(\gamma^C_{[t]}) - \left(\Gamma^{i \to C}_{[t]} \odot \mathbf{K}_{[t]}\right)^\top \mathbf{W}_{[t]}, \quad \mathbf{M} \leftarrow \mathbf{M}_{[t]} \, \mathbf{M}$$

**Merge (forward direction):**

For rank $r$ with `pre_num_ranks` previous ranks:

$$\mathbf{S} = \mathbf{0}; \quad \text{for } j \text{ from } (r - n_\text{pre}) \text{ to } (r-1): \quad \mathbf{S} \leftarrow \mathbf{M}_j \, \mathbf{S} + \mathbf{S}_{\text{ext},j}$$

### Pre-Process Backward

Same structure but **reversed** direction — merges from ranks **after** current rank.

**Stage 1 — $\mathrm{d}\mathbf{S}_\text{ext} \in \mathbb{R}^{d_k \times d_v}$:**

$$\mathrm{d}\mathbf{S} = \mathbf{0}$$

For each sub-chunk $[t]$ (reverse order):

$$\mathrm{d}\mathbf{S} \leftarrow \text{Diag}(\gamma^C_{[t]}) \, \mathrm{d}\mathbf{S}$$
$$\mathrm{d}\mathbf{V} = \mathbf{K}_{[t]} \, \mathrm{d}\mathbf{S} + \mathrm{d}\mathbf{V}_\text{local}$$
$$\mathrm{d}\mathbf{S} \leftarrow \mathrm{d}\mathbf{S} + \left(\Gamma^{1 \to C}_{[t]} \odot \mathbf{Q}_{[t]}\right)^\top \mathrm{d}\mathbf{O}_{[t]} \cdot s - \mathbf{W}_{[t]}^\top \, \mathrm{d}\mathbf{V}$$

where $s = d_k^{-1/2}$ is the scaling factor.

**Stage 2 — $\mathrm{d}\mathbf{M} \in \mathbb{R}^{d_k \times d_k}$:**

$$\mathrm{d}\mathbf{M} = \mathbf{I}$$

For each sub-chunk $[t]$ (reverse order):

$$\mathrm{d}\mathbf{M}_{[t]} = \text{Diag}(\gamma^C_{[t]}) - \mathbf{W}_{[t]}^\top \left(\Gamma^{i \to C}_{[t]} \odot \mathbf{K}_{[t]}\right)$$
$$\mathrm{d}\mathbf{M} \leftarrow \mathrm{d}\mathbf{M}_{[t]} \, \mathrm{d}\mathbf{M}$$

Note: $\mathrm{d}\mathbf{M}_{[t]}$ is the transpose of the forward $\mathbf{M}_{[t]}$ ($\mathbf{W}^\top \mathbf{K}$ vs. $\mathbf{K}^\top \mathbf{W}$).

**Merge (backward direction):**

For rank $r$ with `post_num_ranks` following ranks:

$$\mathrm{d}\mathbf{S} = \mathbf{0}; \quad \text{for } j \text{ from } (r + n_\text{post}) \text{ down to } (r+1): \quad \mathrm{d}\mathbf{S} \leftarrow \mathrm{d}\mathbf{M}_j \, \mathrm{d}\mathbf{S} + \mathrm{d}\mathbf{S}_{\text{ext},j}$$

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
# kg = K ⊙ exp2(γ^{r→C}_{[t]})   rows of Γ^{i→C} ⊙ K
# qg = Q ⊙ exp2(γ^r_{[t]})       rows of Γ^{1→C} ⊙ Q (saved for backward)

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
# qg = Q ⊙ exp2(γ^r_{[t]})
# kg = K ⊙ exp2(γ^{r→C}_{[t]})

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

| Function | GDN | KDA | Gate Path |
|----------|-----|-----|-----------|
| pre_process_fwd | k=$\mathbf{k}$, g=g | k=`kg`, gk=g | GDN: `USE_G`, KDA: `USE_GK` |
| fwd_h | k=$\mathbf{k}$, g=g | k=`kg`, gk=g | Same as pre_process |
| pre_process_bwd | q=$\mathbf{q}$, k=$\mathbf{k}$, g=g | q=`qg`, k=`kg`, gk=g | GDN: `USE_G`, KDA: `USE_GK` |
| bwd_dhu | q=$\mathbf{q}$, k=$\mathbf{k}$, g=g | q=`qg`, k=`kg`, gk=g | Same as pre_process |

**Key consistency**: pre_process and main kernel always receive the **same** tensors.
For KDA, both receive pre-gated `kg` ($= \Gamma^{i \to C}_{[t]} \odot \mathbf{K}_{[t]}$) and `qg` ($= \Gamma^{1 \to C}_{[t]} \odot \mathbf{Q}_{[t]}$).
For GDN, both receive original $\mathbf{k}$, $\mathbf{q}$ (gating applied inside the kernel).

---

## Transition Matrix $\mathbf{M}$

The transition matrix captures how the state transforms across a chunk. Derived from Eq. 8:

$$\mathbf{M}_{[t]} = \text{Diag}(\gamma^C_{[t]}) - \left(\Gamma^{i \to C}_{[t]} \odot \mathbf{K}_{[t]}\right)^\top \mathbf{W}_{[t]} \quad \text{(forward)}$$

$$\mathrm{d}\mathbf{M}_{[t]} = \text{Diag}(\gamma^C_{[t]}) - \mathbf{W}_{[t]}^\top \left(\Gamma^{i \to C}_{[t]} \odot \mathbf{K}_{[t]}\right) \quad \text{(backward, transposed)}$$

Where $\text{Diag}(\gamma^C_{[t]})$ is:
- GDN: $\exp(g_\text{last}) \cdot \mathbf{I}$ — scalar times identity
- KDA: $\text{Diag}(\gamma^C_{[t]})$ — per-dim diagonal, where $\gamma^C_{[t]} = \exp_2(\texttt{gk\_last})$ in code

Cross-rank state is computed by chaining $\mathbf{M}$ matrices:

$$\mathbf{S}_r = \mathbf{M}_{r-1} \left(\mathbf{M}_{r-2} \left(\cdots \mathbf{S}_{\text{ext},0} + \mathbf{S}_{\text{ext},1}\right) + \cdots\right) + \mathbf{S}_{\text{ext},r-1}$$

**Precision note**: The $\mathbf{M}$ chain multiply must stay in fp32 to avoid accumulated precision loss. In bf16, repeatedly casting fp32 accumulators back to bf16 between iterations causes significant error growth over many chunks.

---

## compress_h0 / expand_h0

Optimization for CP mode. Since only the first sequence in the local batch can be a continuation from a previous rank, only its initial state $\mathbf{S}_0 \in \mathbb{R}^{H \times d_k \times d_v}$ is non-zero.
`compress_h0` extracts just that one state to save memory during `save_for_backward`.
`expand_h0` restores the full `[N, H, d_k, d_v]` tensor in backward.
