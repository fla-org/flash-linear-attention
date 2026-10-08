# FLA Environment Variables

Set environment variables before starting Python. Boolean flags accept `0` / `1` unless noted otherwise.

- [Operator backend dispatch](#operator-backend-dispatch)
- [Convolution backend](#convolution-backend)
- [Numerical precision](#numerical-precision)
- [Hardware acceleration](#hardware-acceleration)
- [Autotune & config cache](#autotune--config-cache)
- [Benchmarking](#benchmarking)
- [Testing & CI](#testing--ci)

## Operator backend dispatch

These switches control registered implementations. Unavailable backends and unsupported calls use the default implementation.

| Variable                       | Default           | Options   | Description                                                                                         |
| ------------------------------ | ----------------- | --------- | --------------------------------------------------------------------------------------------------- |
| `FLA_DISABLE_BACKEND_DISPATCH` | `0`               | `0` / `1` | Bypass all backend dispatch. Takes precedence over every backend switch below.                      |
| `FLA_TILELANG`                 | Backend-dependent | `0` / `1` | Enable TileLang implementations when installed. Default enablement depends on the operator and GPU. |
| `FLA_FLASH_KDA`                | `1`               | `0` / `1` | Enable [FlashKDA](https://github.com/MoonshotAI/FlashKDA) for KDA inference when installed.         |

### Gluon

| Variable            | Default | Options   | Description                                                                     |
| ------------------- | ------- | --------- | ------------------------------------------------------------------------------- |
| `FLA_GLUON`         | `0`     | `0` / `1` | Enable all available Gluon backends, overriding individual switches set to `0`. |
| `FLA_ATTNRES_GLUON` | `0`     | `0` / `1` | Enable Gluon AttnRes independently.                                             |

When `FLA_GLUON` is unset or `0`, the individual switches apply. `FLA_DISABLE_BACKEND_DISPATCH=1` overrides all of them.

### Intra-card context parallelism

| Variable                   | Default | Options     | Description                                                                                          |
| -------------------------- | ------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `FLA_INTRACARD_CP`         | `0`     | `0` / `1`   | Enable intra-card CP for shared delta-rule operations; inference and variable-length sequences only. |
| `FLA_INTRACARD_MAX_SPLITS` | `32`    | Integer ≥ 1 | Maximum sub-sequences per original sequence.                                                         |
| `FLA_INTRACARD_TF32X3`     | `0`     | `0` / `1`   | Use TF32x3 for affine-chain dot products on NVIDIA GPUs.                                             |

## Convolution backend

| Variable           | Default | Options           | Description                                                                                                                    |
| ------------------ | ------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `FLA_CONV_BACKEND` | Unset   | `cuda` / `triton` | Override the `backend` argument of `ShortConvolution`, whose default is `triton`. CUDA requires the `causal-conv1d` extension. |

## Numerical precision

| Variable             | Default | Options                    | Description                                                                    |
| -------------------- | ------- | -------------------------- | ------------------------------------------------------------------------------ |
| `FLA_TRIL_PRECISION` | `ieee`  | `ieee` / `tf32` / `tf32x3` | FP32 dot-product precision in `solve_tril`. TF32 modes require NVIDIA support. |
| `FLA_USE_FAST_OPS`   | `0`     | `0` / `1`                  | Enable faster, less accurate math intrinsics in shared operator helpers.       |

## Hardware acceleration

| Variable          | Default | Options                                   | Description                                                                                                    |
| ----------------- | ------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `FLA_USE_TMA`     | `0`     | `0` / `1`                                 | Allow TMA kernels on NVIDIA Hopper/Blackwell and AMD gfx1250, subject to Triton support and input constraints. |
| `FLA_USE_COMPILE` | `1`     | `0` / `1`, `true` / `false`, `yes` / `no` | Enable `torch.compile` for RWKV7 fused addcmul. Case-insensitive; disabled on Python < 3.11.                   |

## Autotune & config cache

| Variable                   | Default       | Options                         | Description                                                       |
| -------------------------- | ------------- | ------------------------------- | ----------------------------------------------------------------- |
| `FLA_CACHE_MODE`           | `disabled`    | See [Cache modes](#cache-modes) | Select how FLA reads pre-tuned kernel configs.                    |
| `FLA_CACHE_RESULTS`        | `1`           | `0` / `1`                       | Persist Triton autotune results when supported.                   |
| `FLA_CONFIG_DIR`           | Unset         | Directory path                  | Read configs from this directory instead of `fla/configs/{GPU}/`. |
| `FLA_GPU_NAME`             | Auto-detected | GPU name                        | Select the GPU config sub-directory.                              |
| `FLA_DISABLE_TENSOR_CACHE` | `0`           | `0` / `1`                       | Disable memoization of helpers with tensor inputs.                |
| `FLA_TENSOR_CACHE_SIZE`    | `4`           | Integer ≥ 0                     | Maximum cached results per tensor-cache helper.                   |

### Cache modes

| Mode       | Behavior                                                 |
| ---------- | -------------------------------------------------------- |
| `disabled` | Skip FLA config lookup and run Triton autotune.          |
| `strict`   | Use an exact-key match; otherwise run autotune.          |
| `fuzzy`    | Try exact and fuzzy matches; otherwise run autotune.     |
| `full`     | Try exact, fuzzy and top-level `default_config` matches. |
| `default`  | Use only the top-level `default_config`.                 |
| `always`   | Use `default_config`, re-reading the JSON on every call. |

## Benchmarking

These variables affect benchmark scripts only.

| Variable                    | Default | Description                                                               |
| --------------------------- | ------- | ------------------------------------------------------------------------- |
| `FLA_BENCH_OP_WARMUP_ITERS` | `5`     | Extra forward/backward warmup iterations per shape.                       |
| `FLA_BENCH_WARMUP_MS`       | `25`    | Triton benchmark warmup duration in milliseconds.                         |
| `FLA_BENCH_REP_MS`          | `100`   | Triton benchmark measurement duration in milliseconds.                    |
| `FLA_BENCH_COOLDOWN_SEC`    | `0`     | Seconds between HEAD and BASE runs in `scripts/run_benchmark_compare.py`. |

## Testing & CI

| Variable     | Default | Options   | Description                                                                |
| ------------ | ------- | --------- | -------------------------------------------------------------------------- |
| `FLA_CI_ENV` | `0`     | `0` / `1` | Allow warnings for small numerical mismatches in `assert_close` during CI. |
