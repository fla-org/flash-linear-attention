# -*- coding: utf-8 -*-

import functools
import os
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch
import triton
from packaging import version


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
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


@lru_cache(maxsize=None)
def is_triton_shared_mem_enough(max_shared_mem: int = 102400, tensor_idx: int = 0) -> bool:
    max_shared_memory = triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['max_shared_mem']
    return max_shared_memory >= max_shared_mem


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['multiprocessor_count']


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        import warnings
        warnings.warn(('Triton is not supported on current platform, roll back to CPU.'), stacklevel=1)
        return 'cpu'


@lru_cache(maxsize=None)
def _check_platform() -> Literal['nvidia', 'amd', 'intel', 'musa']:
    device = get_available_device()
    if device == 'cuda' and 'NVIDIA' in torch.cuda.get_device_name(0):
        return 'nvidia'
    elif device == 'cuda' and 'AMD' in torch.cuda.get_device_name(0):
        return 'amd'
    else:
        return device


device = get_available_device()
device_capacity = is_triton_shared_mem_enough()
device_torch_lib = getattr(torch, device)
device_platform = _check_platform()
is_intel_a770 = (device_platform == 'intel' and 'Intel(R) Arc(TM) A' in torch.xpu.get_device_name(0))
is_nvidia = (device_platform == 'nvidia')
use_cuda_graph = (is_nvidia and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1')

# Nvidia Ampere or newer, haven't check AMD and intel yet.
is_tf32_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8)


def set_torch_device(index: int):
    device_torch_lib.set_device(index)


def contiguous(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            device_index = tensor.device.index
            if device_index is not None:
                set_torch_device(device_index)
        else:
            device_index = 0
            set_torch_device(device_index)

        try:
            result = fn(*contiguous_args, **contiguous_kwargs)
            return result
        finally:
            set_torch_device(0)

    return wrapper


if check_pytorch_version('2.4'):
    device = 'cuda' if device == 'cpu' else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)
else:
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd


def autocast_contiguous_custom_device_fwd(
    fn: callable
) -> callable:
    """
    A decorator that combines the functionality of contiguous and autocast.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_fn = contiguous(fn)
        autocast_contiguous_fn = autocast_custom_fwd(contiguous_fn)
        return autocast_contiguous_fn(*args, **kwargs)
    return wrapper


def autocast_contiguous_custom_device_bwd(
    fn: callable
) -> callable:
    """
    A decorator that combines the functionality of contiguous and autocast.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_fn = contiguous(fn)
        autocast_contiguous_fn = autocast_custom_bwd(contiguous_fn)
        return autocast_contiguous_fn(*args, **kwargs)
    return wrapper
