# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

import pytest
import torch

import fla.utils as fu


def test_assert_close_accepts_close_tensors_and_rejects_mismatch():
    ref = torch.tensor([1.0, 2.0, 3.0])
    tri = ref + 1e-7

    fu.assert_close('close', ref, tri, 1e-3)

    with pytest.raises(AssertionError, match='mismatch'):
        fu.assert_close('mismatch', ref, ref + 1.0, 1e-3)


def test_tensor_cache_uses_identity_until_disabled(monkeypatch):
    calls = {'count': 0}

    @fu.tensor_cache
    def cached_add_one(x):
        calls['count'] += 1
        return x + 1

    x = torch.tensor([1.0])
    y = cached_add_one(x)

    assert cached_add_one(x) is y
    assert calls['count'] == 1

    z = x.clone()
    assert cached_add_one(z) is not y
    assert calls['count'] == 2

    monkeypatch.setattr(fu, 'FLA_DISABLE_TENSOR_CACHE', True)
    assert cached_add_one(x) is not y
    assert calls['count'] == 3


def test_input_guard_makes_tensors_contiguous_except_skipped():
    @fu.input_guard
    def guarded(x, y=None):
        return x, y

    x = torch.randn(2, 3).t()
    y = torch.randn(4, 5).t()
    guarded_x, guarded_y = guarded(x, y)

    assert guarded_x.is_contiguous()
    assert guarded_y.is_contiguous()

    @fu.input_guard(no_guard_contiguous=['y'])
    def skip_y(x, y=None):
        return x, y

    skipped_x, skipped_y = skip_y(x, y)
    assert skipped_x.is_contiguous()
    assert not skipped_y.is_contiguous()

    @fu.input_guard(no_guard_contiguous=('y',))
    def skip_y_with_tuple(x, y=None):
        return x, y

    skipped_x_tuple, skipped_y_tuple = skip_y_with_tuple(x, y)
    assert skipped_x_tuple.is_contiguous()
    assert not skipped_y_tuple.is_contiguous()


def test_deprecate_kwarg_renames_old_keyword_and_warns():
    @fu.deprecate_kwarg('old_value', version='9.9.0', new_name='new_value')
    def fn(new_value=None):
        return new_value

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        assert fn(old_value=42) == 42

    assert len(records) == 1
    assert issubclass(records[0].category, FutureWarning)
    assert 'old_value' in str(records[0].message)


def test_deprecate_kwarg_can_raise_when_both_names_are_set():
    @fu.deprecate_kwarg('old_value', version='9.9.0', new_name='new_value', raise_if_both_names=True)
    def fn(new_value=None):
        return new_value

    with pytest.raises(ValueError, match='Both `old_value` and `new_value`'):
        fn(old_value=1, new_value=2)


def test_public_utils_are_reexported():
    for name in (
        'assert_close',
        'autotune_cache_kwargs',
        'check_shared_mem',
        'device',
        'device_platform',
        'input_guard',
        'is_nvidia_hopper',
        'tensor_cache',
    ):
        assert hasattr(fu, name), name
