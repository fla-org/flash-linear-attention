# Debugging Kimi Context Parallel (KCP) precision failures

A skill-level guide to root-causing KCP precision bugs in `fla`. Read this
before you start chasing a failing `test_cp_*.py`; the patterns here apply
to every op that plugs into KCP (gated-delta-rule, generalized delta rule,
RWKV-style recurrences, future ops).

## TL;DR playbook

1. **Check if distribution is even involved.** Run the failing config in
   a manual KCP simulator (no `torch.distributed`, single device, per-rank
   loop that calls the kernels directly). If it reproduces, NCCL is not
   the culprit and you can iterate without spawning workers.
2. **Compare per-token, not per-chunk.** When KCP and non-KCP chunk the
   same sequence at different boundaries, per-chunk `h`/`dh` mismatches
   are expected and meaningless. Per-token tensors (`v_new`, intermediate
   `dv` from `bwd_dhu`, and the final input-gradients) are the only ones
   that are semantically comparable.
3. **Compare against the same reference twice:**
   - `triton non-KCP` vs `triton KCP` — isolates the KCP path only.
   - `triton non-KCP` vs `naive` (per-token recurrence) — baseline
     chunked-vs-per-token error you will always eat.
   If your KCP path is already close to non-KCP but the real test still
   fails, the error lives somewhere else (forward recomputation, saved
   state, wrapper plumbing…). Do not optimize the kernels.
4. **Ablate H, D, chunk_size, and var-length one at a time.** Uniform
   single-sequence (`--lengths T`) should always be bit-perfect; any
   diff there is a kernel bug, not a KCP-semantic issue.

## Why per-chunk comparisons lie for var-length KCP

For `lengths=[400, 624]`, `chunk_size=64`, `world_size=2`:

- Non-KCP chunks sequence 1 at global tokens `400, 464, 528, 592, ...`.
- Rank 0 chunks its local seq-1 slice `[0, 112)` at local offsets
  `0, 64`, i.e. global `400, 464` (truncated chunk 1 = 48 tokens).
- Rank 1 chunks its local seq-1 slice `[0, 512)` at local offsets
  `0, 64, 128, ...`, i.e. global `512, 576, 640, ...`.

Non-KCP and rank 1 have no common chunk boundaries past 464. The
per-chunk `h[chunk_i]` entries represent state at DIFFERENT tokens,
so comparing them elementwise is nonsense. Per-token tensors still
match because the mathematical recurrence is per-token.

**Rule:** only use per-chunk comparisons when the KCP cut falls on a
chunk boundary for every sequence (e.g. uniform single-sequence or
`lengths=[256, 768]` where every seq start and the rank cut are
multiples of `chunk_size`).

## Semantics of `h` and `dh` across ranks

Both are stored **at the start** of each chunk (state going into the
chunk). So `nocp_h[chunk_i]` = state at token `chunk_i * chunk_size`
within the sequence. For backward, `dh[chunk_i]` is the state gradient
at the same boundary.

In KCP:

- Rank `r`'s fwd merge produces `initial_state` for its FIRST local
  sequence — this is non-KCP's `h` at the first token rank `r` owns.
- Rank `r`'s bwd merge produces `dht` for its LAST local sequence —
  this is non-KCP's `dh` at the token just past rank `r`'s last chunk.

If you are comparing a merged state to non-KCP, line up the global
token index yourself. Don't trust chunk indices.

## Common pitfalls

### 1. Compressed `initial_state` lost across `save_for_backward`

Several ops call `compress_h0(initial_state)` after the forward to
shrink the saved state from `[N_local, H, K, V]` to `[1, H, K, V]`
before backward. If the forward helper only mutates `initial_state`
in its local scope and returns just `(o, final_state, ...)`, the
autograd function saves the ORIGINAL input (`None` for KCP), the
backward recomputation runs `fwd_h(initial_state=None)`, and rank-1+
silently drops the merged state — all downstream per-token gradients
diverge at the 3–5% level.

**Always return the updated `initial_state` from the forward helper
and unpack it in the autograd function before `ctx.save_for_backward`.**
Cross-check with an op that is known to work (e.g. the gated-delta-rule
forward returns `(g, o, A, final_state, initial_state, g_input)` and
its autograd function saves the RETURNED `initial_state`, not the
original argument).

### 2. `expand_h0` order in backward

`expand_h0` must run **before** the forward recomputation in backward,
not after the recomputation right before the backward pre-process.
Otherwise the forward recomputation will index past the compressed
`[1, H, K, V]` buffer for non-first local sequences and read whatever
garbage the torch allocator left behind (often zeros, which masks the
bug for single-sequence ranks but explodes as soon as a rank owns
more than one local sub-sequence).

### 3. autotune `BV` in `merge_fwd_bwd_kernel`

`merge_fwd_bwd_kernel` autotunes `BV ∈ {32, 64}`. Never hardcode `BV`
in a manual grid function — compute it at launch time:

```python
BK = triton.next_power_of_2(K)
def grid(meta): return (triton.cdiv(V, meta['BV']), HV)
merge_fwd_bwd_kernel[grid](..., BK=BK, ...)
```

Hardcoding `BV=64` produces `(cdiv(V, 64), HV)` grid; if the autotuner
picks `BV=32`, the kernel silently only fills half of the `V`
dimension. Symptom: suspiciously huge `dh` diff in a debug simulator
even though the real wrapper works.

### 4. `cu_seqlens` slicing in KCP pre-process

The fwd pre-process uses `cu_seqlens[-2:]` (last local sub-sequence —
the one whose tail gets passed to `rank+1`). The bwd pre-process uses
`cu_seqlens[:2]` (first local sub-sequence — the one whose head
receives `dht` from `rank-1`). These slices are ONE-sub-sequence
windows; the kernel is run with `MULTI_SEQS=False` and does not know
about the other local sequences.

For a rank that owns both a sequence tail and a sequence head (e.g.
rank 2 in CP4 with `lengths=[700, 324]`), the forward and backward
pre-processes handle DIFFERENT sub-sequences. Do not conflate them
when dumping offsets.

### 5. Don't delete `~/.triton/cache` mid-run

Triton compiles lazily and races with running kernels. Wiping the
cache during a live process causes `FileNotFoundError` in the middle
of a kernel launch. Let it live; it's harmless.

### 6. Pytest buffers stdout until the test finishes

Running `pytest -s` still buffers per-test output until the test
returns. For long multi-minute KCP tests this looks like a hang. Run
the test function directly (`python -c 'from tests.x import t; t()'`)
if you need progressive output.

## Debug-script layout

When chasing a new KCP bug, stand up a simulator that mirrors the real
autograd function but runs every rank on a single device. Keep the
structure:

1. `run_nocp(...)` — full non-KCP triton reference (fwd + bwd).
2. `run_cp(...)` — per-rank loop that calls each kernel directly:
   `fwd_intra → wy (if any) → fwd_pre_process → merge → fwd_h →
   bwd_dAu → bwd_pre_process → merge → bwd_dhu → bwd_dv → bwd_o →
   bwd_wy → bwd_dqk_intra`.
3. A pure-torch per-sequence reference via the op's `naive.py` for
   ground truth.

The `run_cp` simulator is the fastest way to localize a bug — you can
print intermediate tensors freely and iterate without paying the
`mp.spawn` + NCCL setup cost on every run. Once the simulator matches
the naive reference bit-by-bf16-bit, you can trust the kernels and
move the investigation to the autograd wrapper (saved tensors,
`compress_h0` / `expand_h0` ordering, `cu_seqlens` plumbing, etc.).

## Acceptance bar

Var-length KCP with `safe_gate=True`, bf16 inputs, and an unaligned
cut point should land below 5e-3 `norm_ratio` per gradient against
the per-token `naive` reference (measured per-sequence). That is pure
bf16 chunk-vs-per-token noise and matches the magnitudes that the
long-standing KCP tests (e.g. gated-delta-rule CP2) sit at. Anything
above ~5e-3 means either the forward recomputation is using the wrong
`initial_state`, or the merge kernel is being called with a stale
`BV`, or both.
