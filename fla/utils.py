# -*- coding: utf-8 -*-

import contextlib
import functools
import inspect
import logging
import os
import sys
import warnings
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Tuple

import torch
import triton
from packaging import version

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fla import __version__

FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"


@lru_cache(maxsize=1)
def check_environments():
    """
    Checks the current operating system, Triton version, and Python version,
    issuing warnings if they don't meet recommendations.
    This function's body only runs once due to lru_cache.
    """
    # Check Operating System
    if sys.platform == 'win32':
        logger.warning(
            "Detected Windows operating system. Triton does not have an official Windows release, "
            "thus FLA will not be adapted for Windows, and any potential errors will not be fixed. "
            "Please consider using a Linux environment for compatibility."
        )

    triton_version = version.parse(triton.__version__)
    required_triton_version = version.parse("3.2.0")

    if triton_version < required_triton_version:
        logger.warning(
            f"Current Triton version {triton_version} is below the recommended 3.2.0 version. "
            "Errors may occur and these issues will not be fixed. "
            "Please consider upgrading Triton."
        )

    # Check Python version
    py_version = version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    required_py_version = version.parse("3.11")

    if py_version < required_py_version:
        logger.warning(
            f"Current Python version {py_version} is below the recommended 3.11 version. "
            "It is recommended to upgrade to Python 3.11 or higher for the best experience."
        )

    return None


check_environments()


def get_abs_err(x, y):
    return (x.detach()-y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach()-y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            import warnings
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg


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


def input_guard(
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
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


contiguous = input_guard


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


class Action(Enum):
    NONE = "none"
    NOTIFY = "notify"
    NOTIFY_ALWAYS = "notify_always"
    RAISE = "raise"


def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: Optional[str] = None,
    warn_if_greater_or_equal_version: bool = False,
    raise_if_greater_or_equal_version: bool = False,
    raise_if_both_names: bool = False,
    additional_message: Optional[str] = None,
):
    """
    Decorator to notify users about deprecated keyword arguments, replacing them with a new name if specified.

    This decorator allows you to:
    - Notify users when a keyword argument is deprecated.
    - Automatically replace deprecated keyword arguments with new ones.
    - Raise an error if deprecated arguments are used, depending on the specified conditions.

    By default, the decorator notifies the user about the deprecated argument while the `fla.__version__` < specified `version`
    in the decorator. To keep notifications with any version `warn_if_greater_or_equal_version=True` can be set.

    Args:
        old_name (`str`):
            Name of the deprecated keyword argument.
        version (`str`):
            The version in which the keyword argument was (or will be) deprecated.
        new_name (`Optional[str]`, *optional*):
            The new name for the deprecated keyword argument.
            If specified, the deprecated keyword argument will be replaced with this new name.
        warn_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to show warning if current `fla` version is greater or equal to the deprecated version.
        raise_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if current `fla` version is greater or equal to the deprecated version.
        raise_if_both_names (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if both deprecated and new keyword arguments are set.
        additional_message (`Optional[str]`, *optional*):
            An additional message to append to the default deprecation message.

    Raises:
        ValueError:
            If `raise_if_greater_or_equal_version` is `True` and the current version >= the deprecated one,
            or if `raise_if_both_names` is `True` and both old and new keyword arguments are provided.

    Returns:
        Callable:
            A wrapped function that handles the deprecated keyword arguments according to the specified parameters.

    Example usage with renaming argument:

        ```python
        @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="6.0.0")
        def my_function(do_reduce_labels):
            print(do_reduce_labels)

        my_function(reduce_labels=True)  # Will show a deprecation warning and use do_reduce_labels=True
        ```

    Example usage without renaming argument:

        ```python
        @deprecate_kwarg("max_size", version="6.0.0")
        def my_function(max_size):
            print(max_size)

        my_function(max_size=1333)  # Will show a deprecation warning
        ```

    """
    deprecated_version = version.parse(version)
    current_version = version.parse(__version__)
    is_greater_or_equal_version = current_version >= deprecated_version

    if is_greater_or_equal_version:
        version_message = f"and removed starting from version {version}"
    else:
        version_message = f"and will be removed in version {version}"

    def wrapper(func):
        # Required for better warning message
        sig = inspect.signature(func)
        function_named_args = set(sig.parameters.keys())
        is_instance_method = "self" in function_named_args
        is_class_method = "cls" in function_named_args

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            # Get class + function name (just for better warning message)
            func_name = func.__name__
            if is_instance_method:
                func_name = f"{args[0].__class__.__name__}.{func_name}"
            elif is_class_method:
                func_name = f"{args[0].__name__}.{func_name}"

            minimum_action = Action.NONE
            message = None

            # deprecated kwarg and its new version are set for function call -> replace it with new name
            if old_name in kwargs and new_name in kwargs:
                minimum_action = Action.RAISE if raise_if_both_names else Action.NOTIFY_ALWAYS
                message = (
                    f"Both `{old_name}` and `{new_name}` are set for `{func_name}`. "
                    f"Using `{new_name}={kwargs[new_name]}` and ignoring deprecated `{old_name}={kwargs[old_name]}`."
                )
                kwargs.pop(old_name)

            # only deprecated kwarg is set for function call -> replace it with new name
            elif old_name in kwargs and new_name is not None and new_name not in kwargs:
                minimum_action = Action.NOTIFY
                message = (
                    f"`{old_name}` is deprecated {version_message} for `{func_name}`. "
                    f"Use `{new_name}` instead."
                )
                kwargs[new_name] = kwargs.pop(old_name)

            # deprecated kwarg is not set for function call and new name is not specified -> just notify
            elif old_name in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message} for `{func_name}`."

            if message is not None and additional_message is not None:
                message = f"{message} {additional_message}"

            # update minimum_action if argument is ALREADY deprecated (current version >= deprecated version)
            if is_greater_or_equal_version:
                # change to (NOTIFY, NOTIFY_ALWAYS) -> RAISE if specified
                # in case we want to raise error for already deprecated arguments
                if raise_if_greater_or_equal_version and minimum_action != Action.NONE:
                    minimum_action = Action.RAISE

                # change to NOTIFY -> NONE if specified (NOTIFY_ALWAYS can't be changed to NONE)
                # in case we want to ignore notifications for already deprecated arguments
                elif not warn_if_greater_or_equal_version and minimum_action == Action.NOTIFY:
                    minimum_action = Action.NONE

            # raise error or notify user
            if minimum_action == Action.RAISE:
                raise ValueError(message)
            elif minimum_action in (Action.NOTIFY, Action.NOTIFY_ALWAYS):
                # DeprecationWarning is ignored by default, so we use FutureWarning instead
                warnings.warn(message, FutureWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper


def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)
    return wrapper


@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


def _cpu_device_warning():
    import warnings
    warnings.warn(('Triton is not supported on current platform, roll back to CPU.'), stacklevel=1)


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['multiprocessor_count']
    except BaseException:
        # Maybe we use a NPU device.
        if triton.runtime.driver.active.get_current_target().backend == 'npu':
            return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['num_vectorcore']
        else:
            return 1


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return 'cpu'


@lru_cache(maxsize=None)
def _check_platform() -> Literal['nvidia', 'amd', 'intel', 'musa', 'npu']:
    device = get_available_device()
    if device == 'cuda':
        return 'nvidia'
    elif device == 'hip':
        return 'amd'
    elif device == 'xpu':
        return 'intel'
    else:
        return device


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = get_available_device() if get_available_device() != 'hip' else 'cuda'
device_torch_lib = getattr(torch, device)
device_platform = _check_platform()

is_amd = (device_platform == 'amd')
is_intel = (device_platform == 'intel')
is_nvidia = (device_platform == 'nvidia')
is_intel_alchemist = (is_intel and 'Intel(R) Arc(TM) A' in torch.xpu.get_device_name(0))
is_nvidia_hopper = (is_nvidia and ('NVIDIA H' in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9))
use_cuda_graph = (is_nvidia and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1')

# Nvidia Ampere or newer, haven't check AMD and intel yet.
is_tf32_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8)
is_gather_supported = hasattr(triton.language, 'gather')


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]


class Backend(Enum):
    ADA = 101376       # RTX 4090
    AMPERE = 166912    # A100
    HOPPER = 232448    # H100
    DEFAULT = 102400   # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


if check_pytorch_version('2.4'):
    device = 'cuda' if device == 'cpu' else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)
else:
    assert device == 'cuda', 'Only cuda device is supported for PyTorch version < 2.4.0.'
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)
