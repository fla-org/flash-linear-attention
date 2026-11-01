# The Design Space of Sparse Attention

**NSA**[^nsa], **MoBA**[^moba], **DSA**[^dsa], and **MSA**[^msa] are *natively-trainable* sparse-attention
mechanisms. They all follow one recipe and differ in only a couple of choices — this note makes that
structure explicit, then defines each precisely.

## The shared recipe

Full causal attention costs `O(T²)` — every query reads every key:

```math
\boldsymbol{o}_t = \sum_{i \le t} \frac{\exp(\boldsymbol{q}_t^\top \boldsymbol{k}_i / \sqrt{d})}{\sum_{j \le t} \exp(\boldsymbol{q}_t^\top \boldsymbol{k}_j / \sqrt{d})}\, \boldsymbol{v}_i
```

Sparse attention attends over a small, **query-dependent** selected set `S_t ⊆ {1,…,t}` instead:

```math
\boldsymbol{o}_t = \sum_{i \in \mathcal{S}_t} \mathrm{softmax}_i\!\big(\boldsymbol{q}_t^\top \boldsymbol{k}_i / \sqrt{d}\big)\, \boldsymbol{v}_i,
\qquad |\mathcal{S}_t| = O(k) \ll t
```

Every method builds `S_t` the same way — **score all candidates with a cheap proxy → take `top-k` →
attend** — because scoring with full attention would cost the `O(T²)` you are trying to avoid. So the
proxy is the whole game, and it is fixed by **two choices**:

- **Granularity** — score and select individual **tokens**, or contiguous **blocks** of keys.
- **Scoring proxy** — rank candidates by **pooled existing keys**, or by a **separate learned indexer**.

Those two choices place the four methods in a 2×2:

|           | **pooled keys** (gradient-free) | **learned indexer** (KL-trained) |
| --------- | ------------------------------- | -------------------------------- |
| **block** | NSA, MoBA                       | MSA                              |
| **token** | —                               | DSA                              |

Each axis has one consequence.

**Granularity sets the quality–speed trade-off.**
Token-level selection is the ideal:
each query picks its own `top-k` tokens, the closest sparse match to full attention.
But token reads are scattered and memory-bound.
Blocks fix that — contiguous, tensor-core- and KV-cache-friendly — at the cost of coarser selection.
So the axis runs **DSA → NSA/MSA → MoBA**, most precise to fastest.

**Scoring sets the training difficulty.**
`top-k` is non-differentiable, so the score has to be learned some other way:

- **Pooled keys** (NSA, MoBA) get gradient *for free*:
  the same keys feed the output, so the LM loss trains them.
- **A separate indexer** (DSA, MSA) feeds *only* selection, never the output, so it gets no gradient.
  It must be **distilled to the full attention** (KL loss: indexer = student, attention = teacher).

**Beyond the two axes,** two patterns recur.
One is a **local + global** split: a cheap local path next to the sparse global one.
The other is a **prefill/decode asymmetry**:
a method sparse in training may still go dense at generation, as MoBA does.

---

## The four instances

Notation: blocks `B_1,…,B_m` of size `ℓ`; `blk(t)` is the block holding token `t`; `k̄_b` is a block's
mean key.

```math
\bar{\boldsymbol{k}}_b = \tfrac{1}{\ell}\sum_{i \in B_b} \boldsymbol{k}_i
```

### NSA — Native Sparse Attention

Paper: [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) — *block · pooled keys*

**Idea.** Three branches, fused by per-token, per-head sigmoid gates (`g_cmp`, `g_slc`, `g_swa`):

```math
\boldsymbol{o}_t = g^{\text{cmp}}_t\, \underbrace{\mathrm{Attn}(\boldsymbol{q}_t, \bar{\boldsymbol{K}}, \bar{\boldsymbol{V}})}_{\text{pooled blocks}}
    + g^{\text{slc}}_t\, \underbrace{\mathrm{Attn}(\boldsymbol{q}_t, \boldsymbol{K}_{\mathcal{I}_t}, \boldsymbol{V}_{\mathcal{I}_t})}_{\text{selected blocks}}
    + g^{\text{swa}}_t\, \underbrace{\mathrm{Attn}(\boldsymbol{q}_t, \boldsymbol{K}_{(t-w,\,t]}, \boldsymbol{V}_{(t-w,\,t]})}_{\text{sliding window}}
```
```math
\mathcal{I}_t = \mathrm{Top}\text{-}n_b\big\{ \boldsymbol{q}_t^\top \bar{\boldsymbol{k}}_b \big\}, \qquad g^{\cdot}_t = \sigma(\cdot) \in (0,1)
```

- **Three branches** — compression (mean-pooled blocks, coarse global), selection (a few whole
  blocks), sliding window (recent tokens).
- **No separate scorer** — selection reuses the compression scores `I_t`, so one coarse pass both
  adds context *and* steers selection (the gradient-free proxy).

**Hardware-aligned core — why GQA is mandatory.**
Selection is *per KV head, not per query head*,
so all `G = H_Q/H_KV` query heads in a group share one block list.
The kernel makes this a GEMM: load the group's queries once as `[G, d]`,
then for each selected block load its KV tile `[d, B_S]` from HBM **once** and matmul against all `G` rows.
Group sharing amortizes each KV read over `G` heads, and block-level reads stay contiguous —
together they restore the arithmetic intensity that scattered, memory-bound sparse attention throws away.

### MoBA — Mixture of Block Attention

Paper: [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) — *block · pooled keys*

**Idea.** NSA's selection branch alone: treat blocks as MoE "experts" and route each query to its `top-k`.

```math
\mathcal{S}_t = \mathrm{Top}\text{-}k_b\big\{ \boldsymbol{q}_t^\top \bar{\boldsymbol{k}}_b : b \le \mathrm{blk}(t) \big\} \cup \{\mathrm{blk}(t)\},
\qquad
\boldsymbol{o}_t = \mathrm{Attn}\!\Big(\boldsymbol{q}_t,\ \{\, (\boldsymbol{k}_i, \boldsymbol{v}_i) : i \in \!\!\bigcup_{b \in \mathcal{S}_t}\!\! B_b,\ i \le t \,\}\Big)
```

- **Minimal** — no gates, no compression, no window branch; the local block is the window, and `k`
  alone sets sparsity (`k=1` → full attention). Local and selected attention merge via online-softmax.
- **Decode falls back to dense** — routing is defined over query blocks, so single-token generation has
  nothing to amortize; MoBA banks its savings on training and prefill.

### DSA — DeepSeek Sparse Attention

Paper: [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) — *token · learned indexer*

**Idea.** A separate lightweight indexer scores every **token**; attention runs over the `top-k` of them.

```math
I_{t,i} = \sum_{h=1}^{H_I} w_{t,h}\, \mathrm{ReLU}\!\big(\boldsymbol{q}^{I\,\top}_{t,h} \boldsymbol{k}^{I}_{i}\big),
\qquad
\mathcal{S}_t = \mathrm{Top}\text{-}k_i\big\{ I_{t,i} : i \le t \big\}
```

- **Lightning indexer** — few heads, low-rank, FP8; ~an order of magnitude cheaper than the main
  attention, which then sees only the pre-filtered `top-k` tokens (`k ≈ 2048`).
- **Token-granular** — maximally flexible but reads are scattered; runs under MLA's MQA mode (one
  shared latent KV) to stay efficient.
- **Retrofit** — added to a *dense* model by continued training: the indexer is first KL-distilled to
  the dense attention, then the whole model adapts.

**Production lineage — GLM-5 → 5.2.**[^glm] Zhipu's GLM-5 family runs DSA on MLA at scale:

- **GLM-5** — adds DSA to a dense base via the same continued-training recipe.
- **GLM-5.1** — same attention; a hardware/kernel port (Ascend / MindSpore, W4A8), no algorithmic change.
- **GLM-5.2** — **IndexShare**: run the indexer once every **4 layers** and reuse its `top-k` indices in
  the next 3 (~2.9× fewer indexer ops at 1M context). A fifth lever the axes above miss — *reuse the
  selection across layers* instead of recomputing it per layer.

### MSA — MiniMax Sparse Attention

Paper: [arXiv:2606.13392](https://arxiv.org/abs/2606.13392) — *block · learned indexer*

**Idea.** Fill the empty quadrant: **DSA's learned indexer, max-pooled to blocks, selected NSA-style.**

```math
S^{g}_{t,b} = \max_{i \in B_b} I^{g}_{t,i},
\qquad
\mathcal{S}^{g}_t = \mathrm{Top}\text{-}k_b\big\{ S^{g}_{t,b} \big\} \cup \{\mathrm{blk}(t)\}
```

- **Pattern = DSA scorer + max-pool + NSA skeleton** — a separate indexer scores tokens (plain
  dot-product, no ReLU), **max-pools** them to blocks, then selects `top-k` blocks **per GQA group**
  (shared across heads in the group, plus a forced local block) exactly like NSA's selection branch.
- **The contribution is training the indexer from scratch** — since the indexer feeds only selection,
  it is KL-distilled to the main attention online (student → teacher), with three stabilizers so the
  auxiliary loss can't corrupt the backbone: **stop-gradient** on the indexer input, an **indexer
  warm-up** (both branches dense at first), and the **forced local block**.
- **Why bother** — its argument against NSA/MoBA: a pooled-key scorer can't suppress the unselected
  tail (those keys get no gradient), so a dedicated, properly-trained indexer reaches higher, cleaner
  sparsity. KV is kept **uncompressed**.

---

## Side-by-side

|               | **NSA**               | **MoBA**           | **DSA**            | **MSA**                        |
| ------------- | --------------------- | ------------------ | ------------------ | ------------------------------ |
| Granularity   | block (+ compression) | block              | **token**          | block (per GQA group)          |
| Scoring proxy | pooled keys (reused)  | pooled keys        | learned indexer    | learned indexer (max-pooled)   |
| Training      | from scratch          | from scratch       | retrofit / distill | from scratch, KL-aligned index |
| Decode        | sparse                | **dense fallback** | sparse             | sparse                         |

Reading the 2×2 as a story: **NSA** is the full block+pooled design; **MoBA** strips it to a single
routed branch; **DSA** moves to token granularity with a learned, KL-trained indexer; **MSA** brings
that indexer back to block granularity (max-pool) to fill the last quadrant.

## References

[^nsa]: **NSA** — *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* DeepSeek-AI, 2026. arXiv:[2502.11089](https://arxiv.org/abs/2502.11089)
[^moba]: **MoBA** — *MoBA: Mixture of Block Attention for Long-Context LLMs.* Moonshot AI (Kimi), 2026. arXiv:[2502.13189](https://arxiv.org/abs/2502.13189) · [FlashMoBA kernel](https://github.com/mit-han-lab/flash-moba)
[^dsa]: **DSA** — *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models* (DeepSeek Sparse Attention). DeepSeek-AI, 2026. arXiv:[2512.02556](https://arxiv.org/abs/2512.02556)
[^glm]: **GLM-5 / IndexShare** — *GLM-5: from Vibe Coding to Agentic Engineering.* Zhipu AI, 2026. arXiv:[2602.15763](https://arxiv.org/abs/2602.15763) · IndexShare write-up: [GLM-5.2 and IndexShare for Long-Context Sparse Attention](https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html)
[^msa]: **MSA** — *MiniMax Sparse Attention.* MiniMax, 2026. arXiv:[2606.13392](https://arxiv.org/abs/2606.13392)
