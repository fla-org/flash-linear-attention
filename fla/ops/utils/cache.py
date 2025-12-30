import json
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import triton
from packaging import version
from triton.runtime.autotuner import Autotuner

TRITON_ABOVE_3_5_1 = version.parse(triton.__version__) >= version.parse("3.5.1")
TRITON_ABOVE_3_4_0 = version.parse(triton.__version__) >= version.parse("3.4.0")
FLA_ALWAYS_CHECK_CACHE = os.environ.get("FLA_ALWAYS_CHECK_CACHE") == "1"
FLA_DISABLE_CACHE = os.environ.get("FLA_DISABLE_CACHE", "1") == "1"


@lru_cache(maxsize=1)
def get_gpu_info():
    """Get GPU model information.

    This function detects the GPU model and returns a sanitized string identifier.
    It prioritizes FLA_GPU_NAME environment variable if set, then detects from
    available hardware (CUDA, ROCm, Intel GPU, or CPU).
    """
    # Check if GPU name is overridden via environment variable
    gpu_name = None
    # Check if GPU name is overridden via environment variable
    if "FLA_GPU_NAME" in os.environ:
        gpu_name = os.environ["FLA_GPU_NAME"]
    # Try to get device name based on availability
    elif torch.cuda.is_available():
        # Works for both NVIDIA and AMD GPUs (ROCm)
        gpu_name = torch.cuda.get_device_name(0)
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        gpu_name = torch.xpu.get_device_name(0)

    if gpu_name:
        return gpu_name.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_")

    # Default to CPU if no GPU available
    return "cpu"


def get_fla_config_dir() -> Path:
    """Get FLA's configs directory.

    The directory can be overridden by setting the FLA_CONFIG_DIR environment variable.
    If set, configs will be loaded from $FLA_CONFIG_DIR/{GPU}/ instead of the default
    fla/configs/{GPU}/ in the project.
    """
    # Check if custom config dir is set via environment variable
    if "FLA_CONFIG_DIR" in os.environ:
        base_dir = Path(os.environ["FLA_CONFIG_DIR"])
    else:
        # Default: project_dir/fla/configs/
        project_dir = Path(__file__).parent.parent.parent
        base_dir = project_dir / "configs"

    gpu_name = get_gpu_info()
    config_dir = base_dir / gpu_name
    return config_dir


def load_cached_config(kernel_name: str) -> dict[str, Any] | None:
    """
    Load cached best config for a kernel from FLA configs directory.

    This function loads the cached best configuration for a given kernel name
    from fla/configs/{GPU}/{kernel_name}.json. The file should contain only the
    best_config dictionary.

    If the config file is not found or cannot be loaded, a warning is printed
    and None is returned, allowing fallback to Triton's autotune.

    The lookup can be disabled by setting the FLA_DISABLE_CACHE environment variable.

    Args:
        kernel_name: Name of the kernel (e.g., "causal_conv1d_fwd_kernel")

    Returns:
        Best config dictionary or None if not found or disabled
    """
    # Check if cache is disabled via environment variable
    if os.environ.get("FLA_DISABLE_CACHE") == "1":
        return None

    config_dir = get_fla_config_dir()
    config_file = config_dir / f"{kernel_name}.json"

    if not config_file.exists():
        return None

    try:
        with open(config_file) as f:
            config = json.load(f)
        return config
    except Exception as e:
        warnings.warn(f"Error reading config file {config_file}: {e}")
        return None


def find_best_config_by_kernel_name(kernel_name: str) -> dict[str, Any] | None:
    """
    Find best config for a kernel by kernel name only (deprecated).

    This function is kept for backward compatibility. Use load_cached_config() instead.

    Args:
        kernel_name: Name of the kernel

    Returns:
        Best config dictionary or None if not found
    """
    warnings.warn(
        "find_best_config_by_kernel_name() is deprecated. "
        "Use load_cached_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return load_cached_config(kernel_name)


class CachedAutotuner(Autotuner):
    """
    A modified autotuner that loads best config from FLA's config directory.

    This class extends Triton's Autotuner but overrides the run method to
    try loading cached configuration first before falling back to autotune.
    """

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, **kwargs):
        super().__init__(fn, arg_names, configs, key, reset_to_zero, restore_value, **kwargs)
        self.kernel_name = fn.fn.__name__ if hasattr(fn, 'fn') else fn.__name__
        self._fla_cache_checked = bool(FLA_DISABLE_CACHE)

    def run(self, *args, **kwargs):
        if not self._fla_cache_checked:
            self.first_run_hook()
            self._fla_cache_checked = bool(not FLA_ALWAYS_CHECK_CACHE)
        return super().run(*args, **kwargs)

    def first_run_hook(self):
        best_config = load_cached_config(self.kernel_name)

        if best_config is not None:
            kw = best_config.get("kwargs", {})
            num_warps = best_config.get("num_warps", 4)
            num_stages = best_config.get("num_stages", 2)

            if TRITON_ABOVE_3_5_1:
                cfg = triton.Config(
                    kw,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    num_ctas=best_config.get("num_ctas", 1),
                    maxnreg=best_config.get("maxnreg", None),
                    pre_hook=None,
                    ir_override=best_config.get("ir_override", None),
                )
            else:
                cfg = triton.Config(
                    kw,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            self.configs = [cfg]
        else:
            # No cached config found.
            warnings.warn(
                f"No cached config found for kernel '{self.kernel_name}', "
                "falling back to Triton autotune",
                stacklevel=2
            )


def cache_autotune(configs, key=None, prune_configs_by=None, reset_to_zero=None, restore_value=None,
                   pre_hook=None, post_hook=None, warmup=None, rep=None, use_cuda_graph=False,
                   do_bench=None, cache_results=False):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function with FLA config support.

    This decorator extends Triton's autotune to support loading best configurations
    from FLA's config directory (fla/configs/{GPU}/). It searches for cached
    configs by kernel name from files named {kernel_name}.json.

    If a cached best config is found, it will be used directly and autotuning will be
    skipped. If no cache is found, a warning is issued and the decorator falls back
    to normal autotuning.

    .. highlight:: python
    .. code-block:: python

        @fla.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size']  # key is used for fallback autotune
        )
        @triton.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...

    :param configs: a list of :code:`triton.Config` objects (used for fallback autotune)
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger autotune
    :type key: list[str] or None
    :param prune_configs_by: a dict of functions that are used to prune configs
    :param reset_to_zero: a list of argument names whose value will be reset to zero
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after running
    :type restore_value: list[str]
    :param pre_hook: a function to call before the kernel
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function to call after the kernel
    :type post_hook: lambda args, exception
    :param warmup: warmup time for benchmarking (deprecated)
    :type warmup: int
    :param rep: repetition time for benchmarking (deprecated)
    :type rep: int
    :param do_bench: a benchmark function
    :type do_bench: lambda fn, quantiles
    :param cache_results: whether to cache autotune timings to disk (passed to Triton)
    :type cache_results: bool
    """
    # key can be None when we want to use cache only (no fallback autotune)
    if key is None:
        key = []

    def decorator(fn):
        kwargs = {}
        if TRITON_ABOVE_3_4_0:
            kwargs = {"cache_results": cache_results}

        return CachedAutotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value,
                               pre_hook=pre_hook, post_hook=post_hook,
                               prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                               use_cuda_graph=use_cuda_graph, do_bench=do_bench,
                               **kwargs,
                               )

    return decorator
