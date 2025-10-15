# FLA Environment Variables

| Variable             | Default | Options                  | Description                                                                              |
| -------------------- | ------- | ------------------------ | ---------------------------------------------------------------------------------------- |
| `FLA_CONV_BACKEND`   | `cuda`  | `triton` or `cuda`       | Choose the convolution backend. `cuda` is the default and preferred for most cases.      |
| `FLA_USE_TMA`        | `0`     | `0` or `1`               | Set to `1` to enable Tensor Memory Accelerator (TMA) on Hopper or Blackwell GPUs.        |
| `FLA_USE_FAST_OPS`   | `0`     | `0` or `1`               | Enable faster, but potentially less accurate, operations when set to `1`.                |
| `FLA_CACHE_RESULTS`  | `1`     | `0` or `1`               | Whether to cache autotune timings to disk. Defaults to `1` (enabled).                    |
| `FLA_TRIL_PRECISION` | `ieee`  | `ieee`, `tf32`, `tf32x3` | Controls the precision for triangular operations. `tf32x3` is only available on NV GPUs. |
