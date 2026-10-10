# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import functools
import importlib.metadata
import inspect
import logging
import os
import shutil
import sys
from functools import cache, lru_cache
from importlib.util import find_spec
from pathlib import Path

import torch
import triton
from packaging import version as package_version

logger = logging.getLogger(__name__)

FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'

FLA_DISABLE_TENSOR_CACHE = os.getenv('FLA_DISABLE_TENSOR_CACHE', '0') == '1'
try:
    FLA_TENSOR_CACHE_SIZE = int(os.getenv('FLA_TENSOR_CACHE_SIZE', "4"))
except ValueError:
    FLA_TENSOR_CACHE_SIZE = 4


@lru_cache(maxsize=1)
def check_environments():
    """
    Checks the current operating system, Triton version, and Python version,
    issuing warnings if they don't meet recommendations.
    This function's body only runs once due to lru_cache.
    """
    # Check Operating System
    if sys.platform == 'win32':
        # Check if triton-windows is installed
        try:
            from importlib.metadata import PackageNotFoundError, metadata
            metadata('triton-windows')
            # triton-windows is installed, no warning needed
        except PackageNotFoundError:
            logger.warning(
                "Detected Windows operating system. Consider installing triton-windows "
                "(https://github.com/triton-lang/triton-windows) for better compatibility. "
                "Without it, some features may not work correctly.",
            )

    triton_version = package_version.parse(triton.__version__)
    required_triton_version = package_version.parse("3.3.0")

    if triton_version < required_triton_version:
        logger.warning(
            f"Current Triton version {triton_version} is below the recommended 3.3.0 version. "
            "Errors may occur and these issues will not be fixed. "
            "Please consider upgrading Triton.",
        )

    # Check Python version
    py_version = package_version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    required_py_version = package_version.parse("3.11")

    if py_version < required_py_version:
        logger.warning(
            f"Current Python version {py_version} is below the recommended 3.11 version. "
            "It is recommended to upgrade to Python 3.11 or higher for the best experience.",
        )

    return None


check_environments()


@cache
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return package_version.parse(torch.__version__) >= package_version.parse(version_s)


TRITON_ABOVE_3_4_0 = package_version.parse(triton.__version__) >= package_version.parse("3.4.0")
TRITON_ABOVE_3_5_1 = package_version.parse(triton.__version__) >= package_version.parse("3.5.1")
TRITON_ABOVE_3_6_0 = package_version.parse(triton.__version__) >= package_version.parse("3.6.0")
TRITON_ABOVE_3_7_1 = package_version.parse(triton.__version__) >= package_version.parse("3.7.1")
TRITON_ABOVE_3_8_0 = package_version.parse(triton.__version__) >= package_version.parse("3.8.0")

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


@functools.cache
def find_spec_cached(name):
    return find_spec(name)


@functools.cache
def has_usable_nvcc() -> bool:
    """Whether a usable nvcc compiler is available for TileLang's JIT.

    Mirrors the guesses in ``tilelang.env._find_cuda_home`` (env
    CUDA_HOME/CUDA_PATH, nvcc on PATH, the ``nvidia-cuda-nvcc`` wheel,
    /usr/local/cuda), but verifies the nvcc binary actually exists —
    only ``nvidia-cuda-nvcc`` >= 13.0 ships it, the ``-cu12`` variant
    installs just ptxas.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None and (Path(cuda_home) / "bin" / "nvcc").exists():
        return True
    if shutil.which("nvcc") is not None:
        return True
    try:
        files = importlib.metadata.files("nvidia-cuda-nvcc") or []
    except importlib.metadata.PackageNotFoundError:
        files = []
    if any(f.name in ("nvcc", "nvcc.exe") for f in files):
        return True
    if (Path("/usr/local/cuda") / "bin" / "nvcc").exists():
        return True

    logger.info(
        "[FLA Backend] TileLang is installed but no usable nvcc compiler was found; falling back to Triton. "
        "Install a CUDA toolkit or nvidia-cuda-nvcc, or set FLA_TILELANG=0 to disable TileLang explicitly."
    )
    return False
