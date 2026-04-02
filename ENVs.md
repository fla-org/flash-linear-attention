# FLA Environment Variables

| Variable             | Default | Options                  | Description                                                                              |
| -------------------- | ------- | ------------------------ | ---------------------------------------------------------------------------------------- |
| `FLA_CONV_BACKEND`   | `cuda`  | `triton` or `cuda`       | Choose the convolution backend. `cuda` is the default and preferred for most cases.      |
| `FLA_USE_TMA`        | `0`     | `0` or `1`               | Set to `1` to enable Tensor Memory Accelerator (TMA) on Hopper or Blackwell GPUs.        |
| `FLA_USE_FAST_OPS`   | `0`     | `0` or `1`               | Enable faster, but potentially less accurate, operations when set to `1`.                |
| `FLA_CACHE_RESULTS`  | `1`     | `0` or `1`               | Whether to cache autotune timings to disk. Defaults to `1` (enabled).                    |
| `FLA_TRIL_PRECISION` | `ieee`  | `ieee`, `tf32`, `tf32x3` | Controls the precision for triangular operations. `tf32x3` is only available on NV GPUs. |
| `FLA_CONFIG_DIR`     | -       | Any path                 | Override the final config directory for Triton kernel configs. When set, loads configs directly from `$FLA_CONFIG_DIR/`; otherwise FLA uses `fla/configs/{GPU}/`. |
| `FLA_CACHE_MODE`     | `disabled` | `disabled`, `strict`, `fuzzy`, `full`, `default`, `always` | Controls how kernel configs are loaded from FLA's config cache. `disabled`: skip all cache, always autotune (default). `strict`: exact key match only; falls back to Triton autotune if no match. `fuzzy`: exact + fuzzy key match; falls back to Triton autotune if no match. `full`: exact key match → fuzzy key match → `default_config` fallback. `default`: use only the top-level `default_config` field, skip key-based lookup. `always`: like `default` but re-reads config files on every kernel call; useful for debugging (edit `default_config` and the next kernel call picks it up without restarting the process). |
| `FLA_GPU_NAME`       | -       | Any string               | Override the detected GPU name for config directory naming. When set, configs will be stored in `configs/{FLA_GPU_NAME}/` instead of auto-detecting from hardware (CUDA/ROCm). Useful for custom or unsupported devices. |
