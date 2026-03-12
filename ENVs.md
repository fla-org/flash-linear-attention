# FLA Environment Variables

| Variable             | Default | Options                  | Description                                                                              |
| -------------------- | ------- | ------------------------ | ---------------------------------------------------------------------------------------- |
| `FLA_CONV_BACKEND`   | `cuda`  | `triton` or `cuda`       | Choose the convolution backend. `cuda` is the default and preferred for most cases.      |
| `FLA_USE_TMA`        | `0`     | `0` or `1`               | Set to `1` to enable Tensor Memory Accelerator (TMA) on Hopper or Blackwell GPUs.        |
| `FLA_USE_FAST_OPS`   | `0`     | `0` or `1`               | Enable faster, but potentially less accurate, operations when set to `1`.                |
| `FLA_CACHE_RESULTS`  | `1`     | `0` or `1`               | Whether to cache autotune timings to disk. Defaults to `1` (enabled).                    |
| `FLA_TRIL_PRECISION` | `ieee`  | `ieee`, `tf32`, `tf32x3` | Controls the precision for triangular operations. `tf32x3` is only available on NV GPUs. |
| `FLA_CONFIG_DIR`     | -       | Any path                 | Override the default config directory for Triton kernel configs. When set, loads configs from `$FLA_CONFIG_DIR/{GPU}/` instead of `fla/configs/{GPU}/`. |
| `FLA_DISABLE_CACHE`  | `1`     | `0` or `1`               | When set to '1', skip loading cached Triton kernel configurations and force fallback to autotune. Useful for debugging or when cache may be outdated. |
| `FLA_GPU_NAME`       | -       | Any string               | Override the detected GPU name for config directory naming. When set, configs will be stored in `configs/{FLA_GPU_NAME}/` instead of auto-detecting from hardware (CUDA/ROCm). Useful for custom or unsupported devices. |
| `TRITON_CACHE_DIR`   | -       | Any path                 | Override Triton's default cache directory. When set, Triton will use this directory for kernel compilation cache instead of `~/.triton`. |
