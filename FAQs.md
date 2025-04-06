# Triton FAQs and Common Issues

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
pip install -U --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126

# Install Triton nightly
pip uninstall triton pytorch-triton -y
pip install -U triton-nightly --index-url http://pypi.fla-org.com/simple --trusted-host pypi.fla-org.com

# Instal flash-linear-attention
pip install einops ninja datasets transformers numpy
pip uninstall flash-linear-attention && pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention --no-deps

# Optional: Install flash-attention
conda install nvidia/label/cuda-12.6.3::cuda-nvcc
pip install packaging psutil ninja
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
