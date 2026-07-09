# Installation

`fla` ships as two PyPI packages: `fla-core` (kernels in `fla/ops`,
`fla/modules`, `fla/utils`) and `flash-linear-attention` (everything in
`fla/layers` + `fla/models`, plus `fla-core` as a dep). Both follow the same
backend-extras layout.

## Pick a backend

`torch` lives in a backend extra, not in the base deps, so wheel metadata is
the same across backends. The `triton` flavor either ships in the extra
(cuda / cpu / npu) or comes transitively from `torch` when you source it
from the matching PyTorch wheel index (rocm / xpu).

| Backend | Extra    | Wheel index                                | `triton` flavor                                  |
| ------- | -------- | ------------------------------------------ | ------------------------------------------------ |
| CUDA    | `[cuda]` | `https://download.pytorch.org/whl/cu128`   | `triton` (PyPI)                                  |
| ROCm    | `[rocm]` | `https://download.pytorch.org/whl/rocm7.2` | pulled by `torch` (`pytorch-triton-rocm` / `triton-rocm`) |
| XPU     | `[xpu]`  | `https://download.pytorch.org/whl/xpu`     | pulled by `torch` (`pytorch-triton-xpu`)         |
| NPU     | `[npu]`  | `https://triton-ascend.osinfra.cn/pypi/simple` | `triton-ascend`                              |
| CPU     | `[cpu]`  | `https://download.pytorch.org/whl/cpu`     | `triton` (PyPI, import-only)                     |

## From PyPI

CUDA can use a single command since `triton` lives on PyPI:
```sh
pip install flash-linear-attention[cuda]
```

For ROCm / XPU / CPU, do it in two steps so `torch` (and the matching `triton`
flavor that `torch` pulls transitively) come from the PyTorch wheel index
instead of letting the resolver mix and match (pip docs are explicit that
there is no priority across configured indices). This mirrors the
[AMD-recommended pattern](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-triton.html):

```sh
# ROCm
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
pip install flash-linear-attention[rocm]

# XPU
pip install --index-url https://download.pytorch.org/whl/xpu torch
pip install flash-linear-attention[xpu]

# CPU
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install flash-linear-attention[cpu]
```

For nightly torch, swap `whl/<backend>` for `whl/nightly/<backend>` and add `--pre`.

## Ascend NPU

NPUs use [`triton-ascend`](https://github.com/triton-lang/triton-ascend), not
upstream `triton`. Since `triton` is in backend extras (not base deps), the
old "install fla, then `pip uninstall triton`, then install `triton-ascend`"
dance is no longer needed.

```sh
# 1. install CANN 9.0.0 + source set_env.sh
# 2. install torch / torch_npu / triton-ascend, then fla with the npu extra
pip install torch==2.7.1 torch_npu==2.7.1 torchvision==0.22.1
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
pip install flash-linear-attention[npu]
```

The `[npu]` extra pins `torch==2.7.1`, `torch_npu==2.7.1`, `torchvision==0.22.1`,
and `triton-ascend==3.2.1` (CANN 9.0.0 stack tested in CI). Install
`triton-ascend` with `--extra-index-url` as shown above.

## From source

```sh
pip uninstall fla-core flash-linear-attention -y
# CUDA
pip install -U "git+https://github.com/fla-org/flash-linear-attention#egg=flash-linear-attention[cuda]"
# Non-CUDA: install backend torch + triton from the PyTorch index first
# (see the per-backend block above), then run the same git+ install with the
# matching extra ([rocm] / [xpu] / [npu] / [cpu]).
```

Or with submodules:
```sh
git submodule add https://github.com/fla-org/flash-linear-attention.git 3rdparty/flash-linear-attention
ln -s 3rdparty/flash-linear-attention/fla fla
```

## Behavior change vs. pre-v0.5

Before v0.5, `pip install flash-linear-attention` resolved a CUDA-built
`torch` + `triton` from the default PyPI index even on ROCm / XPU / NPU
machines, which silently overlaid the wrong wheels. Now the base install
contains no `torch` / `triton` at all: you pick a backend extra. Bare
`pip install flash-linear-attention` no longer imports.

## Notes

- Already have a working `torch` for your backend? `pip install -e .[rocm]`
  (or the matching extra) leaves it alone because the `torch>=2.7.0` pin is
  satisfied.
- For AMD GPUs the `[rocm]` extra pulls `pytorch-triton-rocm`. For Intel GPUs
  the `[xpu]` extra pulls `pytorch-triton-xpu`. See [FAQs](FAQs.md) for
  backend-specific issues.

## Skipping the dep resolver

`torch` pre-release / `triton-nightly` setups can sidestep resolution
entirely:
```sh
pip install transformers einops
pip uninstall fla-core flash-linear-attention -y
pip install -U --no-deps git+https://github.com/fla-org/flash-linear-attention
```
