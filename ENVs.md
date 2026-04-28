# FLA Environment Variables

This page documents the environment variables that influence FLA's behavior at
runtime. Variables are grouped by what they control:

- [Convolution backend](#convolution-backend)
- [Numerical precision](#numerical-precision)
- [Hardware acceleration](#hardware-acceleration)
- [Operator backend dispatch](#operator-backend-dispatch)
- [Autotune & config cache](#autotune--config-cache)
- [Benchmarking](#benchmarking)
- [Testing & CI](#testing--ci)

> **Boolean convention.** Unless noted otherwise, boolean flags accept `0` / `1`.
> A few flags also accept `true` / `yes` (case-insensitive); this is called out
> in the relevant rows.

---

## Convolution backend

| Variable           | Default | Options            | Description                                                                           |
| ------------------ | ------- | ------------------ | ------------------------------------------------------------------------------------- |
| `FLA_CONV_BACKEND` | `cuda`  | `cuda` / `triton`  | Convolution backend used by `ShortConvolution`, Mamba, Mamba2 and log-linear Mamba2.  |

---

## Numerical precision

| Variable             | Default | Options                  | Description                                                                                                |
| -------------------- | ------- | ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `FLA_TRIL_PRECISION` | `ieee`  | `ieee` / `tf32` / `tf32x3` | Precision used by `solve_tril`. `tf32x3` is NVIDIA-only and gives the best performance / accuracy trade-off on Ampere+. |
| `FLA_USE_FAST_OPS`   | `0`     | `0` / `1`                | Enable faster but less accurate Triton math intrinsics in shared op helpers.                               |

---

## Hardware acceleration

| Variable             | Default | Options    | Description                                                                                                  |
| -------------------- | ------- | ---------- | ------------------------------------------------------------------------------------------------------------ |
| `FLA_USE_TMA`        | `0`     | `0` / `1`  | Enable the Tensor Memory Accelerator (TMA) on Hopper / Blackwell GPUs. Requires Triton with TMA support.     |
| `FLA_USE_CUDA_GRAPH` | `0`     | `0` / `1`  | Use CUDA graphs for selected kernels (NVIDIA only).                                                          |
| `FLA_USE_COMPILE`    | `1`     | `1` / `0`, `true` / `false`, `yes` / `no` | Use `torch.compile` for the RWKV7 fused addcmul path. Auto-disabled on Python < 3.11. |

---

## Operator backend dispatch

FLA can dispatch certain ops to specialized backends (TileLang, FlashKDA, intra-card CP).
Each backend is gated by a single env var below.

| Variable                       | Default     | Options    | Description                                                                                                                              |
| ------------------------------ | ----------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `FLA_DISABLE_BACKEND_DISPATCH` | unset (`0`) | `0` / `1`  | Master switch. Set to `1` to bypass *all* backend dispatch and always use the default Triton implementation. Useful for debugging.       |
| `FLA_TILELANG`                 | unset (`1`) | `0` / `1`  | Enable the TileLang backend when the `tilelang` package is installed. Set to `0` to force the Triton path (e.g. to work around #640).    |
| `FLA_FLASH_KDA`                | unset (`1`) | `0` / `1`  | Enable the [FlashKDA](https://github.com/MoonshotAI/FlashKDA) CUTLASS forward for `chunk_kda` (inference only). Requires `flash_kda`.    |
| `FLA_INTRACARD_CP`             | unset (`0`) | `0` / `1`  | Opt in to the intra-card context-parallel backend for shared delta-rule ops (`chunk_gated_delta_rule_fwd_h`). Inference + varlen only.  |
| `FLA_INTRACARD_MAX_SPLITS`     | `32`        | int ≥ 1    | Max number of sub-sequences per original sequence used by the intra-card CP backend. Caps merge-chain depth to control precision loss.  |

---

## Autotune & config cache

FLA ships pre-tuned Triton configs under `fla/configs/{GPU}/`. The variables
below control how those configs are loaded and where Triton's autotune cache
lives.

| Variable                   | Default      | Options                                                            | Description                                                                                                              |
| -------------------------- | ------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `FLA_CACHE_MODE`           | `disabled`   | `disabled` / `strict` / `fuzzy` / `full` / `default` / `always`    | How kernel configs are looked up in FLA's config cache. See [Cache modes](#cache-modes) below.                          |
| `FLA_CACHE_RESULTS`        | `1`          | `0` / `1`                                                          | Cache Triton autotune timings to disk (passed through to `triton.autotune(cache_results=...)` when supported).            |
| `FLA_CONFIG_DIR`           | unset        | any path                                                           | Override the directory FLA reads kernel configs from. When set, configs are loaded from `$FLA_CONFIG_DIR/` directly instead of `fla/configs/{GPU}/`. |
| `FLA_GPU_NAME`             | unset        | any string                                                         | Override the auto-detected GPU name used to pick the config sub-directory. Useful for unsupported / custom devices.       |
| `FLA_DISABLE_TENSOR_CACHE` | `0`          | `0` / `1`                                                          | Disable the in-process `tensor_cache` decorator (which memoizes the latest result of helpers with tensor inputs).         |

### Cache modes

`FLA_CACHE_MODE` controls how aggressively FLA reuses pre-tuned kernel configs:

| Mode       | Behavior                                                                                                                         |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `disabled` | Skip the FLA cache entirely; always run Triton autotune. (Default.)                                                              |
| `strict`   | Exact-key lookup only; fall back to Triton autotune if no match.                                                                 |
| `fuzzy`    | Exact-key → fuzzy-key lookup; fall back to Triton autotune if no match.                                                          |
| `full`     | Exact-key → fuzzy-key → top-level `default_config` fallback.                                                                     |
| `default`  | Use only the top-level `default_config`; skip key-based lookup.                                                                  |
| `always`   | Like `default`, but re-reads the JSON file on every kernel call. Useful for editing `default_config` without restarting Python.   |

---

## Benchmarking

These variables only affect the scripts under `benchmarks/` and `scripts/`;
they have no effect on the library at runtime.

| Variable                    | Default | Description                                                                                                  |
| --------------------------- | ------- | ------------------------------------------------------------------------------------------------------------ |
| `FLA_BENCH_OP_WARMUP_ITERS` | `5`     | Extra forward+backward warmup iterations per shape, on top of Triton's `do_bench` warmup.                    |
| `FLA_BENCH_WARMUP_MS`       | `25`    | Triton `do_bench` warmup time, in **milliseconds**.                                                          |
| `FLA_BENCH_REP_MS`          | `100`   | Triton `do_bench` measurement window, in **milliseconds**.                                                   |
| `FLA_BENCH_COOLDOWN_SEC`    | `0`     | Sleep (seconds) between HEAD and BASE runs in `scripts/run_benchmark_compare.py`. Reduces thermal bias.      |

---

## Testing & CI

| Variable     | Default | Options    | Description                                                                                                  |
| ------------ | ------- | ---------- | ------------------------------------------------------------------------------------------------------------ |
| `FLA_CI_ENV` | `0`     | `0` / `1`  | Mark the process as running in CI. Loosens numeric assertions in `assert_close` so flaky GPU runs warn instead of failing. |
