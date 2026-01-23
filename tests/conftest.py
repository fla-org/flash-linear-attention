import inspect
import math
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import is_fake

try:
    from torch.compiler import is_compiling
except ImportError:
    def is_compiling():
        return False

from fla.utils import device_torch_lib

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CANARY_VALUE = float('nan')
ALIGNMENT_BYTES = 256
MIN_PADDING_ELEMENTS = 1024


class MemoryGuard:
    def __init__(self):
        self.allocations = []

    def register(self, raw_tensor, start_offset, end_offset, shape, dtype):
        self.allocations.append({
            "raw": raw_tensor,
            "start": start_offset,
            "end": end_offset,
            "shape": shape,
            "dtype": dtype
        })

    def verify_and_clear(self):
        errors = []
        for alloc in self.allocations:
            raw = alloc["raw"]
            if is_fake(raw):
                continue
            start = alloc["start"]
            end = alloc["end"]

            # Check Head
            head_slice = raw[:start]
            if not torch.isnan(head_slice).all():
                errors.append(f"[Head Overwritten] Shape: {alloc['shape']} Dtype: {alloc['dtype']}")

            # Check Tail
            tail_slice = raw[end:]
            if not torch.isnan(tail_slice).all():
                errors.append(f"[Tail Overwritten] Shape: {alloc['shape']} Dtype: {alloc['dtype']}")

        self.allocations.clear()
        if errors:
            pytest.fail("\n".join(errors))


_MEMORY_GUARD = MemoryGuard()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

_ORIGINAL_EMPTY = torch.empty
_ORIGINAL_EMPTY_LIKE = torch.empty_like
_ORIGINAL_NEW_EMPTY = torch.Tensor.new_empty


def _get_element_size(dtype):
    return torch.tensor([], dtype=dtype).element_size()


def _calc_aligned_padding(dtype):
    elem_size = _get_element_size(dtype)
    pad_elements = MIN_PADDING_ELEMENTS
    current_bytes = pad_elements * elem_size
    remainder = current_bytes % ALIGNMENT_BYTES
    if remainder != 0:
        missing_bytes = ALIGNMENT_BYTES - remainder
        missing_elems = math.ceil(missing_bytes / elem_size)
        pad_elements += missing_elems
    return pad_elements


def _calc_contiguous_strides(shape):
    """Calculate strides for a contiguous tensor."""
    # For shape (a, b, c), strides are (b*c, c, 1)
    strides = []
    stride = 1
    for s in reversed(shape):
        strides.append(stride)
        stride *= s
    return tuple(reversed(strides))


def _resolve_shape(args):
    if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
        return tuple(args[0])
    return tuple(args)

# -----------------------------------------------------------------------------
# Patched Implementation
# -----------------------------------------------------------------------------


def _is_called_from_fla():
    """Check if the call is from fla package."""
    frame = inspect.currentframe()
    try:
        # Skip the current frame and go up the call stack
        while frame:
            frame = frame.f_back
            if frame is None:
                break

            if hasattr(frame, 'f_code') and hasattr(frame.f_code, 'co_filename'):
                filename = frame.f_code.co_filename
                # Skip conftest.py frames (where the guarded functions are defined)
                if 'conftest.py' in filename:
                    continue
                # Check if this frame is from a test file
                # Look for 'tests/' or 'test_' in the file path
                if 'tests/' in filename or 'test_' in filename:
                    return False

            module = inspect.getmodule(frame)
            if module and hasattr(module, '__name__'):
                # If call is from fla package, apply guard
                if 'fla' in module.__name__:
                    return True
    finally:
        del frame
    # Default to not guarding if we can't determine
    return False


def _guarded_empty(*args, stride=None, **kwargs):
    req_grad = kwargs.pop('requires_grad', False)

    shape = _resolve_shape(args)
    dtype = kwargs.get('dtype') or torch.get_default_dtype()

    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_EMPTY(*args, **kwargs)

    # Check if this call is from fla package or from test directly
    if not _is_called_from_fla():
        # Direct call from test file - don't guard
        return _ORIGINAL_EMPTY(*args, **kwargs)

    numel = math.prod(shape)
    padding_elements = _calc_aligned_padding(dtype)

    total_len = padding_elements + numel + padding_elements
    kwargs_raw = kwargs.copy()
    if 'size' in kwargs_raw:
        del kwargs_raw['size']

    raw_tensor = _ORIGINAL_EMPTY(total_len, **kwargs_raw)
    raw_tensor.fill_(CANARY_VALUE)

    if stride is None:
        stride = _calc_contiguous_strides(shape)
    user_tensor = raw_tensor.as_strided(
        size=shape,
        stride=stride,
        storage_offset=padding_elements
    )

    if user_tensor.is_floating_point():
        user_tensor.fill_(float('nan'))
    elif user_tensor.is_complex():
        user_tensor.fill_(complex(float('nan'), float('nan')))

    if req_grad:
        user_tensor.requires_grad_(True)

    if user_tensor.is_cuda and not is_fake(user_tensor):
        _MEMORY_GUARD.register(raw_tensor, padding_elements, padding_elements + numel, shape, dtype)

    return user_tensor


def _guarded_empty_like(input, **kwargs):
    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    # Skip guard for non-contiguous tensors to avoid as_strided view issues
    if not input.is_contiguous():
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    if kwargs.get('dtype') is None:
        kwargs['dtype'] = input.dtype
    if kwargs.get('layout') is None:
        kwargs['layout'] = input.layout
    if kwargs.get('device') is None:
        kwargs['device'] = input.device
    if kwargs.get('requires_grad') is None:
        kwargs['requires_grad'] = input.requires_grad

    if kwargs.get('layout') != torch.strided:
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    return _guarded_empty(input.shape, **kwargs)


def _guarded_new_empty(self, *args, **kwargs):
    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_NEW_EMPTY(self, *args, **kwargs)

    # Skip guard for non-contiguous tensors to avoid potential issues
    if not self.is_contiguous():
        return _ORIGINAL_NEW_EMPTY(self, *args, **kwargs)

    if kwargs.get('dtype') is None:
        kwargs['dtype'] = self.dtype
    if kwargs.get('device') is None:
        kwargs['device'] = self.device
    if kwargs.get('layout') is None:
        kwargs['layout'] = self.layout
    if kwargs.get('requires_grad') is None:
        kwargs['requires_grad'] = self.requires_grad
    return _guarded_empty(*args, **kwargs)


@pytest.fixture(scope="function", autouse=True)
def poison_torch_memory(request):
    # Only apply the guard to ops and modules tests
    path = str(request.node.fspath)
    if 'tests/ops/' not in path and 'tests/modules/' not in path:
        yield
        return

    with patch('torch.empty', new=_guarded_empty), \
            patch('torch.empty_like', new=_guarded_empty_like), \
            patch('torch.Tensor.new_empty', new=_guarded_new_empty):
        yield
        if hasattr(device_torch_lib, 'synchronize'):
            device_torch_lib.synchronize()
        _MEMORY_GUARD.verify_and_clear()
