# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import contextlib
import functools
import inspect
import sys
import warnings
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
from packaging import version as package_version

from .. import __version__
from ._config import FLA_DISABLE_TENSOR_CACHE, FLA_TENSOR_CACHE_SIZE
from ._device import custom_device_ctx


class Action(Enum):
    NONE = "none"
    NOTIFY = "notify"
    NOTIFY_ALWAYS = "notify_always"
    RAISE = "raise"


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that memoizes the most recent results of a function call by argument identity.

    The decorator keeps a bounded queue of up to ``FLA_TENSOR_CACHE_SIZE`` (default 4)
    recent ``(args, kwargs, result)`` triples. On each call, every cached entry is checked
    in order; an entry is considered a hit when the positional arg count and kwarg key set
    match and every argument is the *same object* (``is`` identity) as the cached one. On a
    hit the cached result is returned and ``fn`` is skipped; on a miss ``fn`` is invoked and
    the new triple is appended (evicting the oldest when the queue is full).

    Caching is fully bypassed when the ``FLA_DISABLE_TENSOR_CACHE`` environment variable is
    set to ``'1'``.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. Intended for functions whose inputs are tensors
            (or other objects compared by identity) and whose output is a tensor.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of ``fn`` backed by an identity-based bounded cache.
    """
    cached: deque = deque(maxlen=FLA_TENSOR_CACHE_SIZE)

    def cache_disabled() -> bool:
        utils_module = sys.modules.get('fla.utils')
        return getattr(utils_module, 'FLA_DISABLE_TENSOR_CACHE', FLA_DISABLE_TENSOR_CACHE)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if cache_disabled():
            return fn(*args, **kwargs)

        for cached_args, cached_kwargs, cached_result in cached:
            if len(args) != len(cached_args) or len(kwargs) != len(cached_kwargs):
                continue
            if all(a is b for a, b in zip(args, cached_args, strict=False)) and \
                    all(k in cached_kwargs and v is cached_kwargs[k] for k, v in kwargs.items()):
                return cached_result

        result = fn(*args, **kwargs)
        cached.append((args, kwargs, result))
        return result

    return wrapper


def _skip_contiguous(
    no_guard_contiguous: bool | list[str] | tuple[str, ...] | set[str],
    param_name: str,
    skip_params: set[str],
) -> bool:
    return no_guard_contiguous is True or param_name in skip_params


def _contiguous_if_needed(arg: Any, skip: bool) -> Any:
    if isinstance(arg, torch.Tensor) and not skip:
        return arg.contiguous()
    return arg


def input_guard(
    fn: Callable[..., torch.Tensor] | None = None,
    *,
    no_guard_contiguous: bool | list[str] | tuple[str, ...] | set[str] = False,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]] | Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.

    Args:
        no_guard_contiguous (bool | list[str] | tuple[str, ...] | set[str]):
            If True, skip all contiguous checks. If a list/tuple/set of parameter names, skip contiguous check for those parameters.
    """

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        # Get function signature for parameter name mapping
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        skip_params = set(no_guard_contiguous) if isinstance(no_guard_contiguous, (list, tuple, set)) else set()

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Process args with parameter name mapping
            processed_args = []
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                else:
                    # For *args beyond signature, use position as name
                    param_name = f"__arg_{i}"

                processed_args.append(_contiguous_if_needed(
                    arg, _skip_contiguous(no_guard_contiguous, param_name, skip_params)))

            # Process kwargs
            processed_kwargs = {}
            for k, v in kwargs.items():
                processed_kwargs[k] = _contiguous_if_needed(v, _skip_contiguous(no_guard_contiguous, k, skip_params))

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
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    # Handle direct usage without parentheses: @input_guard
    if fn is not None:
        return decorator(fn)

    return decorator


def contiguous(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Alias for input_guard() without parameters."""
    return input_guard(fn)


def require_version(version, hint):
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version
            require_version(version, hint)
            return fn(
                ctx,
                *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()},
            )
        return wrapper
    return decorator


def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: str | None = None,
    warn_if_greater_or_equal_version: bool = False,
    raise_if_greater_or_equal_version: bool = False,
    raise_if_both_names: bool = False,
    additional_message: str | None = None,
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
    deprecated_version = package_version.parse(version)
    current_version = package_version.parse(__version__)
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
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)
    return wrapper
