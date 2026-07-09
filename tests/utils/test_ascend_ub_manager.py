# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

import pytest
import torch
import triton

import fla.utils.ascend_ub_manager as ub_mod
from fla.utils import IS_NPU
from fla.utils.ascend_ub_manager import (
    _FALLBACK_UB_CAPACITY_BITS,
    ASCEND_MAX_GRID_DIM,
    UBManager,
    _default_strategy,
    compute_activation_block_size,
    compute_default_tiling_strategy,
    compute_elementwise_block_size,
    compute_grid_limited_tile_size,
    compute_row_tile_block_size,
    compute_ub_block_size,
    compute_vocab_block_size,
    get_ub_manager,
    is_npu_available,
    iter_axis_launch_chunks,
    max_grid_axis_chunks,
)

requires_npu = pytest.mark.skipif(
    not IS_NPU,
    reason='Ascend NPU hardware required',
)


@pytest.fixture(autouse=True)
def reset_ub_manager_singleton():
    ub_mod._ub_manager = None
    yield
    ub_mod._ub_manager = None


@pytest.fixture
def ub_capacity_bits():
    return _FALLBACK_UB_CAPACITY_BITS


class TestNormalizeAndDefaultStrategy:
    def test_normalize_tiling_dims_int_and_tuple(self):
        assert ub_mod._normalize_tiling_dims(0) == {0}
        assert ub_mod._normalize_tiling_dims((0, 1)) == {0, 1}
        assert ub_mod._normalize_tiling_dims(()) == set()
        assert ub_mod._normalize_tiling_dims('bad') == set()

    def test_default_strategy_empty_inputs(self):
        assert _default_strategy(
            _FALLBACK_UB_CAPACITY_BITS, 0.8, 4, 10.0, (), ()
        ) == ()
        assert _default_strategy(
            _FALLBACK_UB_CAPACITY_BITS, 0.8, 4, 10.0, ((4096,),), ()
        ) == ()

    def test_default_strategy_invalid_tiling_dim(self):
        with pytest.raises(ValueError, match='Invalid tiling_dim'):
            _default_strategy(
                _FALLBACK_UB_CAPACITY_BITS, 0.8, 4, 10.0, ((4096,),), ((),)
            )
        with pytest.raises(ValueError, match='Invalid tiling_dim'):
            _default_strategy(
                _FALLBACK_UB_CAPACITY_BITS, 0.8, 4, 10.0, ((4096,),), (1,)
            )

    def test_default_strategy_power_of_two_block(self, ub_capacity_bits):
        max_safe = _default_strategy(
            ub_capacity_bits,
            safety_margin=0.80,
            dtype_size=4,
            memory_multiplier=10.0,
            shapes=((4096,),),
            tiling_dims=(0,),
        )
        assert len(max_safe) == 1
        block = max_safe[0]
        assert block >= 1
        assert block & (block - 1) == 0  # power of 2


class TestUBManager:
    def test_explicit_capacity_properties(self):
        manager = UBManager(ub_capacity_bits=524288)
        assert manager.ub_capacity_bits == 524288
        assert manager.ub_capacity_bytes == 65536

    def test_detect_from_env_var(self, monkeypatch):
        monkeypatch.setenv('ASCEND_UB_CAPACITY_BITS', '1048576')
        manager = UBManager()
        assert manager.ub_capacity_bits == 1048576

    def test_invalid_env_var_falls_back(self, monkeypatch):
        monkeypatch.setenv('ASCEND_UB_CAPACITY_BITS', 'not-a-number')
        monkeypatch.setattr(ub_mod, 'is_npu_available', lambda: False)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            manager = UBManager()
        assert manager.ub_capacity_bits == _FALLBACK_UB_CAPACITY_BITS
        assert any('fallback UB capacity' in str(w.message) for w in records)

    def test_npu_unavailable_fallback(self, monkeypatch):
        monkeypatch.delenv('ASCEND_UB_CAPACITY_BITS', raising=False)
        monkeypatch.setattr(ub_mod, 'is_npu_available', lambda: False)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            manager = UBManager()
        assert manager.ub_capacity_bits == _FALLBACK_UB_CAPACITY_BITS
        assert any('NPU is not available' in str(w.message) for w in records)

    def test_get_ub_manager_singleton(self, monkeypatch):
        monkeypatch.setenv('ASCEND_UB_CAPACITY_BITS', '524288')
        first = get_ub_manager()
        second = get_ub_manager()
        assert first is second
        assert first.ub_capacity_bits == 524288


class TestComputeDefaultTilingStrategy:
    def test_returns_none_for_invalid_inputs(self, monkeypatch):
        monkeypatch.setenv('ASCEND_UB_CAPACITY_BITS', str(_FALLBACK_UB_CAPACITY_BITS))
        assert compute_default_tiling_strategy(shapes=None, tiling_dims=(0,)) is None
        assert compute_default_tiling_strategy(shapes=(), tiling_dims=()) is None
        assert compute_default_tiling_strategy(
            shapes=((4096,),), tiling_dims=None
        ) is None
        assert compute_default_tiling_strategy(
            shapes=((4096,),), tiling_dims=(0, 1)
        ) is None

    def test_geglu_like_tiling(self, monkeypatch):
        monkeypatch.setenv('ASCEND_UB_CAPACITY_BITS', str(_FALLBACK_UB_CAPACITY_BITS))
        result = compute_default_tiling_strategy(
            safety_margin=0.80,
            dtype_size=2,
            memory_multiplier=7.0,
            shapes=((4096,),),
            tiling_dims=(0,),
        )
        assert result is not None
        block_size, = result[0]
        assert block_size >= 1
        assert block_size <= triton.next_power_of_2(4096)
        assert block_size & (block_size - 1) == 0

    def test_rope_like_tiling(self, monkeypatch):
        monkeypatch.setenv('ASCEND_UB_CAPACITY_BITS', str(_FALLBACK_UB_CAPACITY_BITS))
        result = compute_default_tiling_strategy(
            safety_margin=0.90,
            dtype_size=4,
            memory_multiplier=3.0,
            shapes=((32, 128), (32, 128)),
            tiling_dims=(0, 0),
        )
        assert result == ((32, 128), (32, 128))

    def test_non_tiling_dims_padded_to_power_of_two(self, monkeypatch):
        monkeypatch.setenv('ASCEND_UB_CAPACITY_BITS', str(_FALLBACK_UB_CAPACITY_BITS))
        result = compute_default_tiling_strategy(
            safety_margin=0.80,
            dtype_size=4,
            memory_multiplier=3.0,
            shapes=((32, 100),),
            tiling_dims=(0,),
        )
        assert result is not None
        _, padded_col = result[0]
        assert padded_col == triton.next_power_of_2(100)

    def test_invalid_tiling_dim_raises(self, monkeypatch):
        monkeypatch.setenv('ASCEND_UB_CAPACITY_BITS', str(_FALLBACK_UB_CAPACITY_BITS))
        with pytest.raises(ValueError, match='Invalid tiling_dim'):
            compute_default_tiling_strategy(
                shapes=((4096,),),
                tiling_dims=((),),
            )


class TestBlockSizeHelpers:
    @pytest.fixture(autouse=True)
    def _fixed_ub(self, monkeypatch):
        monkeypatch.setenv('ASCEND_UB_CAPACITY_BITS', str(_FALLBACK_UB_CAPACITY_BITS))

    def test_compute_ub_block_size_respects_bounds(self):
        block = compute_ub_block_size(
            4096,
            memory_multiplier=10.0,
            min_block=64,
            max_block=512,
        )
        assert 64 <= block <= 512

    def test_compute_ub_block_size_uses_desired_override(self):
        block = compute_ub_block_size(
            4096,
            memory_multiplier=10.0,
            desired=128,
            max_block=2048,
        )
        assert block == 128

    def test_compute_vocab_block_size_respects_grid_limit(self):
        block = compute_vocab_block_size(
            vocab_size=50000,
            num_rows=4,
            memory_multiplier=10.0,
            max_block=8192,
        )
        max_splits = max(1, ASCEND_MAX_GRID_DIM // 4)
        grid_min = triton.next_power_of_2((50000 + max_splits - 1) // max_splits)
        assert block >= grid_min
        assert block <= 8192

    def test_compute_elementwise_block_size(self):
        n_elements = ASCEND_MAX_GRID_DIM * 4
        block = compute_elementwise_block_size(n_elements, memory_multiplier=2.5)
        assert block >= 1024
        assert triton.cdiv(n_elements, block) <= ASCEND_MAX_GRID_DIM

    def test_compute_activation_block_size_forward_and_backward(self):
        total = 1_000_000
        fwd = compute_activation_block_size(total, is_backward=False)
        bwd = compute_activation_block_size(total, is_backward=True)
        assert 256 <= fwd <= 2048
        assert 256 <= bwd <= 2048
        assert bwd <= fwd
        assert triton.cdiv(total, fwd) <= ASCEND_MAX_GRID_DIM

    def test_compute_row_tile_block_size_row_and_col(self):
        row_block = compute_row_tile_block_size(
            row_dim=128,
            fixed_dim=64,
            memory_multiplier=4.0,
            tiling_row=True,
            max_block=128,
        )
        col_block = compute_row_tile_block_size(
            row_dim=128,
            fixed_dim=64,
            memory_multiplier=4.0,
            tiling_row=False,
            max_block=64,
        )
        assert 1 <= row_block <= 128
        assert 1 <= col_block <= 64

    def test_compute_grid_limited_tile_size(self):
        assert compute_grid_limited_tile_size(4096, 32, 1024) == 1024
        assert compute_grid_limited_tile_size(512, 32, 1024) == 512
        assert compute_grid_limited_tile_size(4096, 32, 1024, min_block=2048) == 2048


class TestGridAxisHelpers:
    def test_max_grid_axis_chunks(self):
        assert max_grid_axis_chunks(100000, 1) == ASCEND_MAX_GRID_DIM
        assert max_grid_axis_chunks(100000, 2) == ASCEND_MAX_GRID_DIM // 2
        assert max_grid_axis_chunks(100000, 0) == ASCEND_MAX_GRID_DIM

    def test_iter_axis_launch_chunks_covers_axis(self):
        axis_size = 100_000
        chunks = list(iter_axis_launch_chunks(axis_size, other_grid_product=1))
        assert chunks[0] == (0, ASCEND_MAX_GRID_DIM)
        covered = sum(length for _, length in chunks)
        assert covered == axis_size
        offsets = [offset for offset, _ in chunks]
        assert offsets == sorted(offsets)
        assert len(offsets) == len(set(offsets))

    def test_iter_axis_launch_chunks_single_chunk(self):
        chunks = list(iter_axis_launch_chunks(100, other_grid_product=1))
        assert chunks == [(0, 100)]


class TestIsNpuAvailable:
    def test_returns_bool_without_crashing(self):
        assert isinstance(is_npu_available(), bool)


@requires_npu
class TestAscendNPUIntegration:
    @pytest.fixture(autouse=True)
    def _clear_ub_env(self, monkeypatch):
        monkeypatch.delenv('ASCEND_UB_CAPACITY_BITS', raising=False)

    def test_is_npu_available_on_device(self):
        assert is_npu_available() is True
        assert torch.npu.is_available()

    def test_ub_manager_detects_soc_ub_size(self):
        from tbe.common.platform import get_soc_spec, set_current_compile_soc_info

        soc_info = torch.npu.get_device_name(torch.npu.current_device())
        set_current_compile_soc_info(soc_info)
        ub_size_bytes = get_soc_spec('UB_SIZE')

        manager = UBManager()
        assert ub_size_bytes > 0
        assert manager.ub_capacity_bits == ub_size_bytes * 8
        assert manager.ub_capacity_bytes == ub_size_bytes

    def test_ub_manager_detects_npu_model(self):
        manager = UBManager()
        dev_name = torch.npu.get_device_properties(0).name
        assert manager.npu_model == dev_name
        assert manager.npu_model != 'unknown'

    def test_get_ub_manager_singleton_uses_detected_capacity(self):
        from tbe.common.platform import get_soc_spec, set_current_compile_soc_info

        soc_info = torch.npu.get_device_name(torch.npu.current_device())
        set_current_compile_soc_info(soc_info)
        expected_bits = get_soc_spec('UB_SIZE') * 8

        manager = get_ub_manager()
        assert manager.ub_capacity_bits == expected_bits

    def test_env_var_overrides_soc_detection(self, monkeypatch):
        override_bits = 524288
        monkeypatch.setenv('ASCEND_UB_CAPACITY_BITS', str(override_bits))
        manager = UBManager()
        assert manager.ub_capacity_bits == override_bits

    def test_compute_default_tiling_strategy_with_real_ub(self):
        result = compute_default_tiling_strategy(
            safety_margin=0.80,
            dtype_size=2,
            memory_multiplier=7.0,
            shapes=((4096,),),
            tiling_dims=(0,),
        )
        assert result == ((4096,),)

        rope = compute_default_tiling_strategy(
            safety_margin=0.90,
            dtype_size=4,
            memory_multiplier=3.0,
            shapes=((32, 128), (32, 128)),
            tiling_dims=(0, 0),
        )
        assert rope == ((32, 128), (32, 128))

    def test_compute_ub_block_size_with_real_ub(self):
        block = compute_ub_block_size(4096, memory_multiplier=10.0)
        assert block == triton.next_power_of_2(4096)

    def test_compute_activation_block_size_with_real_ub(self):
        total = 1_000_000
        fwd = compute_activation_block_size(total, is_backward=False)
        bwd = compute_activation_block_size(total, is_backward=True)
        assert fwd == 2048
        assert bwd == 2048
        assert triton.cdiv(total, fwd) <= ASCEND_MAX_GRID_DIM

    def test_real_ub_exceeds_fallback_capacity(self):
        manager = UBManager()
        assert manager.ub_capacity_bits > _FALLBACK_UB_CAPACITY_BITS
