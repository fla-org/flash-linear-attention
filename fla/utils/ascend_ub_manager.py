# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# copied from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/backends/_ascend/ub_manager.py

"""
Unified Buffer (UB) Manager for Ascend NPU.

This module provides UB capacity detection and tiling strategy computation
for running Triton kernels on Ascend NPU. It automatically calculates
optimal block sizes based on UB capacity constraints to prevent UB overflow.
"""

import os
import warnings

import torch
import triton

# Ascend Triton launch limits (see triton_ascend/activations.py).
ASCEND_MAX_GRID_DIM = 65535
# Legacy fused kernel byte cap (65536 // fp16 element size).
_FALLBACK_MAX_FUSED_BLOCK = 65536 // 2
# Conservative UB fallback when runtime detection is unavailable (64 KiB).
_FALLBACK_UB_CAPACITY_BITS = 65536 * 8


def is_npu_available() -> bool:
    """Detect Ascend NPU availability."""
    if hasattr(torch, 'npu'):
        try:
            return torch.npu.is_available()
        except Exception:
            pass
    try:
        from transformers.utils import is_torch_npu_available

        return is_torch_npu_available()
    except ImportError:
        return False


def _fallback_ub_capacity(reason: str) -> int:
    warnings.warn(
        f"Using fallback UB capacity ({_FALLBACK_UB_CAPACITY_BITS // 8} bytes): {reason}",
        stacklevel=3,
    )
    return _FALLBACK_UB_CAPACITY_BITS


def _normalize_tiling_dims(tiling_dim: int | tuple[int, ...]) -> set:
    """
    Normalize tiling dimension specification to a set of dimension indices.

    Args:
        tiling_dim: Either an int (single dimension) or tuple of ints (multiple dimensions).

    Returns:
        Set of dimension indices that can be tiled.
    """
    if isinstance(tiling_dim, int):
        return {tiling_dim}
    elif isinstance(tiling_dim, tuple):
        return set(tiling_dim)
    else:
        return set()


def _default_strategy(
    ub_capacity_bits: int,
    safety_margin: float,
    dtype_size: int,
    memory_multiplier: float,
    shapes: tuple[tuple[int, ...], ...],
    tiling_dims: tuple[int | tuple[int, ...], ...],
) -> tuple[int, ...]:
    """
    Default tiling strategy: calculate maximum safe block size based on UB capacity.

    This is a unified strategy function that works for all kernels by abstracting
    the memory calculation as: memory_multiplier * BLOCK_SIZE * unit_param * dtype_size * 8 bits

    Args:
        ub_capacity_bits: UB capacity in bits
        safety_margin: Safety margin as a float (e.g., 0.80 for 80%)
        dtype_size: Size of data type in bytes (e.g., 2 for float16, 4 for float32)
        memory_multiplier: Memory multiplier for estimating peak memory usage
        shapes: Tuple of full shapes. Each shape is a tuple of dimension sizes.
            - For ROPE: ((n_q_head, hd), (n_kv_head, hd))
            - For GEGLU: ((n_cols,),)
        tiling_dims: Tuple specifying which dimensions can be tiled for each shape.
            Each element can be:
            - int: single dimension index (e.g., 0 for first dimension)
            - tuple of ints: multiple dimensions that can be tiled together
            - For ROPE: (0, 0) means first dimension of each shape can be tiled
            - For GEGLU: (0,) means first dimension of the shape can be tiled
            Length must match len(shapes).

    Returns:
        Tuple of maximum safe block sizes, one for each shape.
        Each element is a power of 2.

    Note:
        For each shape, fixed dimensions (non-tiling) are multiplied together to get unit_param.
        The final block size is computed in compute_default_tiling_strategy by taking
        min(desired_block_size, max_safe_block_size) where desired_block_size = triton.next_power_of_2(original_dim).
    """
    if not shapes or not tiling_dims:
        return ()

    # Calculate max_safe_block_size for each tiling dimension
    max_safe_sizes = []

    for shape, tiling_dim in zip(shapes, tiling_dims):
        # Normalize tiling_dim to a set of dimension indices
        tiling_dim_set = _normalize_tiling_dims(tiling_dim)

        # Validate tiling dimensions are within shape bounds
        if not tiling_dim_set:
            raise ValueError(
                f"Invalid tiling_dim: {tiling_dim}. tiling_dim must be an int or a non-empty tuple of ints."
            )
        if any(dim_idx < 0 or dim_idx >= len(shape) for dim_idx in tiling_dim_set):
            raise ValueError(
                f"Invalid tiling_dim: {tiling_dim} for shape {shape}. "
                f"All dimension indices must be in range [0, {len(shape)})."
            )

        # Calculate unit_param: product of fixed (non-tiling) dimensions
        unit_param = 1.0
        for dim_idx, dim_size in enumerate(shape):
            if dim_idx not in tiling_dim_set:
                if dim_size <= 0:
                    # Invalid dimension size, use conservative default
                    unit_param = 1.0
                    break
                unit_param *= float(dim_size)

        # Ensure unit_param is at least 1.0
        if unit_param <= 0:
            unit_param = 1.0

        # Calculate maximum safe block size based on UB capacity
        # Memory: memory_multiplier * BLOCK_SIZE * unit_param * dtype_size * 8 bits
        SAFE_UB_CAPACITY_BITS = int(ub_capacity_bits * safety_margin)

        # Solve: memory_multiplier * BLOCK_SIZE * unit_param * dtype_size * 8 <= SAFE_UB_CAPACITY_BITS
        # BLOCK_SIZE <= SAFE_UB_CAPACITY_BITS / (memory_multiplier * unit_param * dtype_size * 8)
        max_block_size = int(SAFE_UB_CAPACITY_BITS // (memory_multiplier * unit_param * dtype_size * 8))
        max_block_size = max(1, max_block_size)

        # Find largest power of 2 <= max_block_size
        # Use triton.next_power_of_2(max_block_size + 1) // 2 to get the largest power of 2 <= max_block_size
        safe_block_size = triton.next_power_of_2(max_block_size + 1) // 2
        max_safe_sizes.append(safe_block_size)

    return tuple(max_safe_sizes)


class UBManager:
    """
    Unified Buffer Manager for Ascend NPU.

    Provides UB capacity detection and management for Ascend NPU devices.
    The UB capacity is used by tiling strategy functions to calculate optimal block sizes.
    """

    def __init__(self, ub_capacity_bits: int | None = None):
        """
        Initialize UB Manager.

        Args:
            ub_capacity_bits: UB capacity in bits. If None, will be detected automatically.
        """
        self._npu_model = self._detect_npu_model()
        self._ub_capacity_bits = ub_capacity_bits or self._detect_ub_capacity()

    @property
    def ub_capacity_bits(self) -> int:
        """Get UB capacity in bits."""
        return self._ub_capacity_bits

    @property
    def ub_capacity_bytes(self) -> int:
        """Get UB capacity in bytes."""
        return self._ub_capacity_bits // 8

    @property
    def npu_model(self) -> str:
        """Get detected NPU model name."""
        return self._npu_model

    def _detect_npu_model(self) -> str:
        """Detect NPU model from device properties."""
        if not is_npu_available():
            return "unknown"

        try:
            dev_props = torch.npu.get_device_properties(0)
            # Try to get model name from device properties
            return dev_props.name
        except Exception:
            pass

        return "default"

    def _detect_ub_capacity(self) -> int:
        """
        Detect UB capacity from environment variable or get_soc_spec.

        Returns:
            UB capacity in bits. Falls back to a conservative default with a warning
            when detection fails.
        """
        # Check environment variable first (in bits)
        env_capacity = os.getenv("ASCEND_UB_CAPACITY_BITS")
        if env_capacity is not None:
            try:
                capacity_bits = int(env_capacity)
                if capacity_bits > 0:
                    return capacity_bits
            except ValueError:
                pass

        # Try to get from get_soc_spec (returns bytes, convert to bits)
        if is_npu_available():
            try:
                from tbe.common.platform import get_soc_spec, set_current_compile_soc_info

                # Set current SOC info for get_soc_spec to work correctly
                device = torch.npu
                soc_info = device.get_device_name(device.current_device())
                set_current_compile_soc_info(soc_info)

                # Query UB size (get_soc_spec returns size in bytes)
                ub_size_bytes = get_soc_spec("UB_SIZE")

                if ub_size_bytes is None or ub_size_bytes <= 0:
                    raise ValueError(f"Invalid UB_SIZE from get_soc_spec: {ub_size_bytes}")

                # Convert bytes to bits
                ub_capacity_bits = ub_size_bytes * 8
                return ub_capacity_bits

            except ImportError:
                return _fallback_ub_capacity(
                    "Cannot import tbe.common.platform.get_soc_spec. "
                    "Source CANN set_env.sh or set ASCEND_UB_CAPACITY_BITS."
                )
            except Exception as e:
                return _fallback_ub_capacity(
                    f"Failed to detect UB capacity from get_soc_spec: {e}. "
                    "Set ASCEND_UB_CAPACITY_BITS to override."
                )

        return _fallback_ub_capacity(
            "NPU is not available. Set ASCEND_UB_CAPACITY_BITS to override."
        )


# Global singleton instance
_ub_manager: UBManager | None = None


def get_ub_manager() -> UBManager:
    """Get global UB manager instance."""
    global _ub_manager
    if _ub_manager is None:
        _ub_manager = UBManager()
    return _ub_manager


def compute_default_tiling_strategy(
    safety_margin: float = 0.80,
    dtype_size: int | None = None,
    memory_multiplier: float | None = None,
    shapes: tuple[tuple[int, ...], ...] | None = None,
    tiling_dims: tuple[int | tuple[int, ...], ...] | None = None,
) -> tuple[tuple[int, ...], ...] | None:
    """
    Compute tiling strategy using the default strategy function.

    This function directly calls the default strategy and computes the final
    tiling result. All kernels use the same unified strategy function, so
    there's no need for kernel_name-based lookup.

    Args:
        safety_margin: Safety margin as a float (e.g., 0.80 for 80%). Default is 0.80.
        dtype_size: Size of data type in bytes (e.g., 2 for float16, 4 for float32).
            Must be provided. If None or <= 0, defaults to 4 (float32).
        memory_multiplier: Memory multiplier for estimating peak memory usage.
            - For GEGLU: typically 10.0 for backward, 4.0 for forward
            - For ROPE: typically 3.0
            If None, defaults to 10.0 (conservative estimate).
        shapes: Tuple of full shapes. Each shape is a tuple of dimension sizes.
            - For ROPE: ((n_q_head, hd), (n_kv_head, hd))
            - For GEGLU: ((n_cols,),)
            Can pass original shapes (will handle padding internally) or padded shapes.
        tiling_dims: Tuple specifying which dimensions can be tiled for each shape.
            Each element can be:
            - int: single dimension index (e.g., 0 for first dimension)
            - tuple of ints: multiple dimensions that can be tiled together
            - For ROPE: (0, 0) means first dimension of each shape can be tiled
            - For GEGLU: (0,) means first dimension of the shape can be tiled
            Length must match len(shapes). Cannot be empty.

    Returns:
        Tuple of tiled shapes with same structure as input shapes.
        Tiling dimensions are replaced with computed block sizes (power of 2),
        while non-tiling dimensions are padded to next power of 2.
        - For ROPE: ((block_size_q, pad_hd), (block_size_kv, pad_hd))
        - For GEGLU: ((block_size,),)
        Returns None if shapes or tiling_dims is None or empty.

    Examples:
        >>> # ROPE forward
        >>> strategy = compute_default_tiling_strategy(
        ...     safety_margin=0.90,
        ...     dtype_size=4,
        ...     memory_multiplier=3.0,
        ...     shapes=((32, 128), (32, 128)),
        ...     tiling_dims=(0, 0)
        ... )
        >>> # Returns: ((block_size_q, 128), (block_size_kv, 128))
        >>> # GEGLU forward
        >>> strategy = compute_default_tiling_strategy(
        ...     safety_margin=0.80,
        ...     dtype_size=2,
        ...     memory_multiplier=7.0,
        ...     shapes=((4096,),),
        ...     tiling_dims=(0,)
        ... )
        >>> # Returns: ((block_size,),)
    """
    ub_manager = get_ub_manager()

    if shapes is None or not shapes or tiling_dims is None or not tiling_dims:
        return None

    if len(shapes) != len(tiling_dims):
        return None

    if dtype_size is None or dtype_size <= 0:
        dtype_size = 4  # Default to float32

    if memory_multiplier is None or memory_multiplier <= 0:
        memory_multiplier = 10.0  # Default conservative estimate

    # Call strategy to get max_safe_block_size for each shape
    max_supported = _default_strategy(
        ub_manager.ub_capacity_bits,
        safety_margin,
        dtype_size,
        memory_multiplier,
        shapes,
        tiling_dims,
    )

    if not max_supported or len(max_supported) != len(shapes):
        return None

    # Build result: same structure as shapes, with tiling dims replaced by computed block sizes
    result = []
    for shape, tiling_dim, max_safe in zip(shapes, tiling_dims, max_supported):
        result_shape = list(shape)

        # Normalize tiling_dim to a set of dimension indices
        tiling_dim_set = _normalize_tiling_dims(tiling_dim)

        # Validate tiling dimensions are within shape bounds
        if not tiling_dim_set:
            raise ValueError(
                f"Invalid tiling_dim: {tiling_dim}. tiling_dim must be an int or a non-empty tuple of ints."
            )
        if any(dim_idx < 0 or dim_idx >= len(result_shape) for dim_idx in tiling_dim_set):
            raise ValueError(
                f"Invalid tiling_dim: {tiling_dim} for shape {shape}. "
                f"All dimension indices must be in range [0, {len(result_shape)})."
            )

        # Replace tiling dimensions with computed block sizes
        # For each tiling dimension, compute: min(desired, max_safe)
        for dim_idx in tiling_dim_set:
            original_dim = result_shape[dim_idx]
            desired = triton.next_power_of_2(original_dim)
            final_val = min(desired, max_safe)
            final_val = max(1, final_val)  # Ensure at least 1
            result_shape[dim_idx] = final_val

        # Pad non-tiling dimensions to next power of 2
        for dim_idx, dim_size in enumerate(result_shape):
            if dim_idx not in tiling_dim_set:
                result_shape[dim_idx] = triton.next_power_of_2(dim_size)

        result.append(tuple(result_shape))

    return tuple(result)


def compute_ub_block_size(
    dim_size: int,
    memory_multiplier: float,
    *,
    safety_margin: float = 0.9,
    dtype_size: int = 4,
    fallback: int = 2048,
    min_block: int = 1,
    max_block: int | None = None,
    desired: int | None = None,
) -> int:
    """Compute UB-safe block size for a single tilable dimension."""
    if desired is None:
        desired = triton.next_power_of_2(dim_size)

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=safety_margin,
        dtype_size=dtype_size,
        memory_multiplier=memory_multiplier,
        shapes=((dim_size,),),
        tiling_dims=(0,),
    )
    block = min(desired, tile_shapes[0][0]) if tile_shapes else min(desired, fallback)
    block = max(min_block, block)
    if max_block is not None:
        block = min(block, max_block)
    return block


def compute_vocab_block_size(
    vocab_size: int,
    num_rows: int,
    memory_multiplier: float,
    *,
    safety_margin: float = 0.9,
    max_block: int = 8192,
    fallback: int = 2048,
) -> int:
    """UB-safe vocab tile size respecting Ascend grid dim0 limit."""
    ub_block = compute_ub_block_size(
        vocab_size,
        memory_multiplier,
        safety_margin=safety_margin,
        fallback=fallback,
    )
    max_splits = max(1, ASCEND_MAX_GRID_DIM // max(num_rows, 1))
    grid_min = triton.next_power_of_2((vocab_size + max_splits - 1) // max_splits)
    return min(max(ub_block, grid_min), max_block)


def compute_elementwise_block_size(
    n_elements: int,
    memory_multiplier: float = 2.5,
    *,
    safety_margin: float = 0.9,
    min_block: int = 1024,
    fallback: int | None = None,
) -> int:
    """UB-safe block size for elementwise kernels under grid limit."""
    if fallback is None:
        fallback = _FALLBACK_MAX_FUSED_BLOCK
    ub_block = compute_ub_block_size(
        n_elements,
        memory_multiplier,
        safety_margin=safety_margin,
        fallback=fallback,
    )
    grid_min = max(
        min_block,
        triton.next_power_of_2((n_elements + ASCEND_MAX_GRID_DIM - 1) // ASCEND_MAX_GRID_DIM),
    )
    return min(ub_block, grid_min)


def compute_activation_block_size(
    total_elements: int,
    is_backward: bool,
    *,
    max_grid: int = ASCEND_MAX_GRID_DIM,
    max_core_dim: int = 65535,
    safety_margin: float = 0.9,
    max_block: int = 2048,
    memory_multiplier: float | None = None,
) -> int:
    """UB-safe block size for flattened activation kernels."""
    if memory_multiplier is None:
        memory_multiplier = 6.0 if is_backward else 3.0
    block = compute_ub_block_size(
        total_elements,
        memory_multiplier,
        safety_margin=safety_margin,
        fallback=2048,
        min_block=256,
        max_block=max_block,
    )
    block = min(block, max_core_dim // 8)
    if triton.cdiv(total_elements, block) > max_grid:
        block = triton.cdiv(total_elements, max_grid)
        block = max(triton.next_power_of_2(block), 1)
    return block


def compute_row_tile_block_size(
    row_dim: int,
    fixed_dim: int,
    memory_multiplier: float,
    *,
    tiling_row: bool = True,
    safety_margin: float = 0.85,
    dtype_size: int = 4,
    fallback: int = 16,
    min_block: int = 1,
    max_block: int | None = None,
) -> int:
    """UB-safe tile along one axis of a 2D [row, col] kernel tile."""
    if tiling_row:
        shapes = ((row_dim, fixed_dim),)
        tiling_dims = (0,)
        desired = triton.next_power_of_2(row_dim)
    else:
        shapes = ((row_dim, fixed_dim),)
        tiling_dims = (1,)
        desired = triton.next_power_of_2(fixed_dim)

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=safety_margin,
        dtype_size=dtype_size,
        memory_multiplier=memory_multiplier,
        shapes=shapes,
        tiling_dims=tiling_dims,
    )
    if tile_shapes:
        block = tile_shapes[0][0] if tiling_row else tile_shapes[0][1]
        block = min(desired, block)
    else:
        block = min(desired, fallback)
    block = max(min_block, block)
    if max_block is not None:
        block = min(block, max_block)
    return block


def max_grid_axis_chunks(
    axis_size: int,
    other_grid_product: int,
    *,
    max_grid: int = ASCEND_MAX_GRID_DIM,
) -> int:
    """Max launch chunks along one grid axis while keeping the product <= max_grid."""
    return max(1, max_grid // max(other_grid_product, 1))


def compute_grid_limited_tile_size(
    axis_size: int,
    other_grid_product: int,
    ub_safe_block: int,
    *,
    max_grid: int = ASCEND_MAX_GRID_DIM,
    min_block: int = 1,
) -> int:
    """Pick a UB-safe tile size; host-side chunking handles grid overflow."""
    return max(min_block, min(ub_safe_block, axis_size))


def iter_axis_launch_chunks(
    axis_size: int,
    other_grid_product: int,
    *,
    max_grid: int = ASCEND_MAX_GRID_DIM,
):
    """Yield ``(offset, chunk_len)`` for host-side grid-axis tiling."""
    max_chunks = max_grid_axis_chunks(axis_size, other_grid_product, max_grid=max_grid)
    for offset in range(0, axis_size, max_chunks):
        yield offset, min(max_chunks, axis_size - offset)
