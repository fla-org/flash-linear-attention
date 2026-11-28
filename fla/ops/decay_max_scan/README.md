# decay_max_scan

Input-dependent decay running-max scan with an O(1) recurrent state.

## Recurrence

Per lane (b, d, h), with input `x_t` and per-channel decay parameters `v_d, b_d`:

```
g_t = exp(-softplus(x_t * v_d + b_d))    # input-dependent decay, g_t in (0, 1)
e_t = max(x_t, g_t * e_{t-1}),           # e_0 = -1e30 (cold start)
```

`e_t` is the decay-weighted running maximum: `e_t = max_{s<=t} x_s * prod_{r=s+1..t} g_r`.
Cold start uses `-1e30` instead of `-inf` so the `d * state` gradient term stays finite.

## Chunked evaluation (chunk.py)

`max(x_s * prod_{r=s+1..t} g_r)` over `s <= t` is a scan under the associative
semiring `(max, *)` with pairs `(m, d)` and

```
combine((m_l, d_l), (m_r, d_r)) = (max(m_r, d_r * m_l), d_r * d_l)
```

so a Hillis-Steele doubling scan (log2(CHUNK) vectorized steps) evaluates each
chunk without `cumprod` division: the naive `cumprod/cummax` formulation
underflows when `prod(g)` reaches fp32 subnormals (e.g. g ~ 0.5 over 128 steps),
which poisons the cummax through `x / pref`. The doubling form only clamps the
decay product at `1e-30`, where the decayed history term is negligible for
bounded inputs.

## Triton backend (triton_backend.py)

Dense CUDA batches dispatch to Triton kernels (`triton_backend.py`): a
block-local reverse scan produces per-chunk carries `(g, m, owner)`, a carry
composition kernel sweeps the sequence writing outputs and the argmax-owner
chain, and the backward is a reverse BPTT kernel over per-position coefficients
recomputed in torch (no coefficient tensors saved for backward). varlen
(`cu_seqlens`) and non-CUDA tensors run the torch doubling scan above.
Measured throughput (RTX 4090 D, B8-32, T384-2048, D96, H8): Triton 18-21 Mtok/s,
torch semiring 0.8-3.8, sequential naive 0.13-0.76. The Triton backend is
validated on Windows (triton 3.2) and Linux (WSL2; triton 3.2 and 3.6 on the
FLA target stack, torch 2.11), including `FLA_CI_ENV=0` strict-mode tests.

## Gradients

Where the input branch wins (`x_t > g_t * e_{t-1}`), gradients flow into `x_t`;
otherwise they flow to `e_{t-1}` scaled by `g_t`, plus the decay-parameter
terms through `g_t = exp(-softplus(x_t * v_d + b_d))` (sigmoid factor). The
chunked scan is differentiable end to end; `tests/ops/test_decay_max_scan.py`
checks outputs, final states, and `dx/dv/db` gradients against `naive.py`.

## Decoding

`fused_recurrent_decay_max_scan` performs the single-token update; the state is
one `[B, D, H]` tensor — strictly O(1) per lane.
