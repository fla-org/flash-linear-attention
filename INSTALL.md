# Installation

`fla` ships as two PyPI packages: `fla-core` (kernels in `fla/ops`,
`fla/modules`, `fla/utils`) and `flash-linear-attention` (everything in
`fla/layers` + `fla/models`, plus `fla-core` as a dep). Both follow the same
backend-extras layout.

## Pick a backend

`torch` and the matching `triton` flavor live in backend extras, not in the
base deps. Wheel metadata is the same across backends; `pip` only pulls the
flavor you ask for.

| Backend | Extra    | Wheel index                                | `triton` flavor          |
| ------- | -------- | ------------------------------------------ | ------------------------ |
| CUDA    | `[cuda]` | `https://download.pytorch.org/whl/cu128`   | `triton`                 |
| ROCm    | `[rocm]` | `https://download.pytorch.org/whl/rocm7.2` | `pytorch-triton-rocm`    |
| XPU     | `[xpu]`  | `https://download.pytorch.org/whl/xpu`     | `pytorch-triton-xpu`     |
| NPU     | `[npu]`  | (use your CANN-matched `torch` / `torch_npu`) | `triton-ascend`       |
| CPU     | `[cpu]`  | `https://download.pytorch.org/whl/cpu`     | `triton` (import-only)   |

## From PyPI

```sh
pip install flash-linear-attention[cuda]
pip install flash-linear-attention[rocm] --extra-index-url https://download.pytorch.org/whl/rocm7.2
pip install flash-linear-attention[xpu]  --extra-index-url https://download.pytorch.org/whl/xpu
pip install flash-linear-attention[cpu]  --extra-index-url https://download.pytorch.org/whl/cpu
```

## Ascend NPU

NPUs use [`triton-ascend`](https://github.com/triton-lang/triton-ascend), not
upstream `triton`. Since `triton` is in backend extras (not base deps), the
old "install fla, then `pip uninstall triton`, then install `triton-ascend`"
dance is no longer needed.

```sh
# 1. install CANN + source set_env.sh
# 2. install torch + torch_npu that match your CANN
pip install torch==2.7.1 torch_npu==2.7.1
# 3. install fla with the npu extra
pip install flash-linear-attention[npu]
```

## From source

```sh
pip uninstall fla-core flash-linear-attention -y
pip install -U "git+https://github.com/fla-org/flash-linear-attention#egg=flash-linear-attention[cuda]"
# replace [cuda] with [rocm] / [xpu] / [npu] / [cpu] and add the matching --extra-index-url for non-CUDA
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
