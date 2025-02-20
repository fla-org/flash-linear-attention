# -*- coding: utf-8 -*-

import functools
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton
from packaging import version
from functools import lru_cache


def contiguous(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous.
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


def tensor_cache(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


def require_version(version, hint):
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version
            require_version(version, hint)
            return fn(ctx,
                      *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                      **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
        return wrapper
    return decorator


def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)
    return wrapper


@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str = "2.4") -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


@lru_cache(maxsize=None)
def check_triton_shared_mem(max_shared_mem: int = 102400, tensor_idx: int = 0) -> bool:
    max_shared_memory = triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['max_shared_mem']
    return max_shared_memory >= max_shared_mem


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['multiprocessor_count']


@lru_cache(maxsize=None)
def get_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except ImportError:
        pass
    if torch.xpu.is_available():
        return "xpu"

    try:
        import torch_musa  # noqa: F401

        if torch.musa.is_available():
            return "musa"
    except ImportError:
        pass

    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu"
    except ImportError:
        pass

    raise ValueError("No available device found.")


device = "cuda" if get_available_device() == "cpu" else get_available_device()
device_capacity = check_triton_shared_mem()
device_torch_lib = getattr(torch, device)
is_intel_a770 = (device == "xpu" and 'Intel(R) Arc(TM) A' in torch.xpu.get_device_name(0))


def set_torch_device(x: torch.Tensor):
    device_torch_lib.set_device(x.device.index)


if check_pytorch_version("2.4"):
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)
else:
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd
