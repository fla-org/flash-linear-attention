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

CUDA can use a single command since `triton` lives on PyPI:
```sh
pip install flash-linear-attention[cuda]
```

For ROCm / XPU / CPU, do it in two steps so pip pulls the backend `torch` +
`triton` flavor from the PyTorch index instead of letting the resolver pick a
mix (pip docs are explicit that there is no priority across configured
indices). Pick your index, install `torch` + the `triton` flavor first, then
install `fla`:

```sh
# ROCm
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch pytorch-triton-rocm
pip install flash-linear-attention[rocm]

# XPU
pip install --index-url https://download.pytorch.org/whl/xpu torch pytorch-triton-xpu
pip install flash-linear-attention[xpu]

# CPU
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install flash-linear-attention[cpu]
```

## Ascend NPU

NPUs use [`triton-ascend`](https://github.com/triton-lang/triton-ascend), not
upstream `triton`. Since `triton` is in backend extras (not base deps), the
old "install fla, then `pip uninstall triton`, then install `triton-ascend`"
dance is no longer needed.

```sh
# 1. install CANN + source set_env.sh
# 2. install fla with the npu extra (pins torch / torch_npu / triton-ascend)
pip install flash-linear-attention[npu]
```

`triton-ascend` 3.2.1 is tagged but not yet on PyPI, so the `[npu]` extra
currently pins `torch==2.6.0`, `torch_npu==2.6.0`, `triton-ascend==3.2.0`.

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
