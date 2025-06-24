# Triton FAQs and Common Issues

* [MMA Assertion](#1-mma-assertion-error-on-h100)
* [AsstibuteError](#2-attributeerror-nonetype-object-has-no-attribute-start)
* [LinearLayout](#3-h100-linearlayout-assertion-error)
* [Triton on Arm](#4-triton-support-for-arm-aarch64-architecture)

## Recommended Setup Approach

> [!IMPORTANT]
> Triton nightly builds often depend on the latest PyTorch nightly versions. To prevent conflicts with existing installations, we strongly recommend creating a fresh conda environment. This isolates the installation from any existing PyTorch/Triton versions that might cause compatibility issues.

## Common Issues and Solutions

### 1. MMA Assertion Error on H100

**Error:**
```py
Assertion `!(srcMmaLayout && dstMmaLayout && !srcMmaLayout.isAmpere()) && "mma -> mma layout conversion is only supported on Ampere"' failed.
```

**Solution:**
This issue was fixed in [PR #4492](https://github.com/triton-lang/triton/pull/4492). Install the nightly version:

```sh
# Create fresh environment (strongly recommended!!!)
conda create -n triton-nightly python=3.12
conda activate triton-nightly

# Install PyTorch nightly (required for Triton nightly compatibility)
pip install -U --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Install Triton nightly
pip uninstall triton pytorch-triton -y
pip install -U triton-nightly --index-url https://pypi.fla-org.com/simple

# Instal flash-linear-attention
pip install einops ninja datasets transformers numpy
pip uninstall flash-linear-attention && pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention --no-deps

# Optional: Install flash-attention
conda install nvidia/label/cuda-12.8.1::cuda-nvcc
pip install packaging psutil ninja
pip install git+https://github.com/Dao-AILab/causal-conv1d.git --no-build-isolation
pip install flash-attn --no-deps --no-cache-dir --no-build-isolation

# Optional: Verify flash-attention installation
pip install pytest
pytest tests/ops/test_attn.py
```

### 2. AttributeError: 'NoneType' object has no attribute 'start'

**Solution:**
This is a known issue ([triton-lang/triton#5224](https://github.com/triton-lang/triton/issues/5224)). Upgrade to Python 3.10+.

### 3. H100 LinearLayout Assertion Error

**Error:**
```
mlir::triton::LinearLayout::reshapeOuts(...) failed.
```

**Solution:**
This is a known issue ([triton-lang/triton#5609](https://github.com/triton-lang/triton/issues/5609)). Follow the same installation steps as in Issue #1 above.

### 4. Triton Support for ARM (aarch64) Architecture
Triton now supports the ARM (aarch64) architecture.

However, official Triton and PyTorch do not provide pre-built binaries for this architecture. The FLA organization has manually built and provided support for Triton on ARM, currently covering Triton versions 3.2.x, 3.3.x, and nightly builds.

**Installation for ARM (aarch64):**

For users on ARM (aarch64) systems, directly installing triton and pytorch from their official channels can be challenging as pre-built binaries for this architecture are often unavailable. The FLA organization provides custom-built Triton binaries to address this, ensuring compatibility with specific PyTorch versions.

To ensure a smooth installation of flash-linear-attention with the necessary Triton and PyTorch dependencies on ARM, it's crucial to align their versions. The FLA builds of Triton are designed to be compatible with particular PyTorch releases.

**Version Compatibility:**

Below is a guide to compatible triton and pytorch versions when using FLA's Triton builds:

- Triton 3.2.0 is compatible with PyTorch 2.6.0
- Triton 3.3.0 is compatible with PyTorch 2.7.0
- Triton 3.3.1 is compatible with PyTorch 2.7.1

```shell
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -U triton==3.3.1 --index-url https://pypi.fla-org.com/simple
```
