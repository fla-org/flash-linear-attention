# FLA Environment Variables

## Compute

| Variable             | Default | Options                  | Description                                                                              |
| -------------------- | ------- | ------------------------ | ---------------------------------------------------------------------------------------- |
| `FLA_CONV_BACKEND`   | `cuda`  | `triton` or `cuda`       | Choose the convolution backend. `cuda` is the default and preferred for most cases.      |
| `FLA_USE_TMA`        | `0`     | `0` or `1`               | Set to `1` to enable Tensor Memory Accelerator (TMA) on Hopper or Blackwell GPUs.        |
| `FLA_USE_FAST_OPS`   | `0`     | `0` or `1`               | Enable faster, but potentially less accurate, operations when set to `1`.                |
| `FLA_TRIL_PRECISION` | `ieee`  | `ieee`, `tf32`, `tf32x3` | Controls the precision for triangular operations. `tf32x3` is only available on NV GPUs. |

## Cache

| Variable         | Default        | Options   | Description                                                                                                                                                              |
| ---------------- | -------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `FLA_CACHE`      | `~/.cache/fla` | Any path  | Base cache directory for FLA. All cache data (configs, etc.) is stored under this path.                                                                                  |
| `FLA_CONFIG_DIR` | -              | Any path  | Override the final config directory for Triton kernel configs. When set, loads configs directly from `$FLA_CONFIG_DIR/`; otherwise FLA uses `$FLA_CACHE/configs/{GPU}/`. |
| `FLA_CACHE_MODE` | `disabled`     | See below | Controls how kernel configs are loaded from FLA's config cache.                                                                                                          |

`FLA_CACHE_MODE` options:

| Mode                | Lookup order                                            | Fallback               |
| ------------------- | ------------------------------------------------------- | ---------------------- |
| `disabled`          | Skip all cache                                          | Always Triton autotune |
| `strict`            | Exact key match                                         | Triton autotune        |
| `fuzzy`             | Exact key → fuzzy match (numeric wildcard)              | Triton autotune        |
| `full`              | Exact key → fuzzy match → `default_config`              | Triton autotune        |
| `default`           | `default_config` only, skip key-based lookup            | Triton autotune        |
| `always`            | Same as `default`, but re-reads config files every call | Triton autotune        |

| Variable            | Default | Options   | Description                                                           |
| ------------------- | ------- | --------- | --------------------------------------------------------------------- |
| `FLA_CACHE_RESULTS` | `1`     | `0` or `1`| Whether to cache autotune timings to disk. Defaults to `1` (enabled). |

## Hardware

| Variable       | Default | Options    | Description                                                                                                                                                                                                                         |
| -------------- | ------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FLA_GPU_NAME` | -       | Any string | Override the detected GPU name for config directory naming. When set, configs will be stored in `$FLA_CACHE/configs/{FLA_GPU_NAME}/` instead of auto-detecting from hardware (CUDA/ROCm). Useful for custom or unsupported devices. |
