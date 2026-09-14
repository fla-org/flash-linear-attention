# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Last-dim contract for Triton-Ascend kernels.

CANN leftover DMA of a blocked tile is rounded to 32 elements.
If the last-dim byte stride is not a multiple of 32, that leftover aliases the next row.

triton_ascend therefore requires ``dim * itemsize % 32 == 0``.
Unsupported last dims fail the backend verifier (and ``npu_require_last_dims``).
Leftover-K with honest ``shape=(T, K)`` plus ``boundary_check`` is zeroed by CANN;
``MASK_LEFTOVER`` is only for partial T tiles and varlen (gate diffs before ``exp2``).
"""

from __future__ import annotations

import math

import torch

NPU_DMA_ALIGN_BYTES = 32


def npu_last_dim_elem_align(dtype: torch.dtype) -> int:
    """Elements per last dim so ``dim * itemsize`` is a multiple of 32 bytes."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    return NPU_DMA_ALIGN_BYTES // math.gcd(itemsize, NPU_DMA_ALIGN_BYTES)


def npu_supported_last_dim(n: int, *, dtype: torch.dtype) -> bool:
    """True when the last-dim byte stride is a multiple of ``NPU_DMA_ALIGN_BYTES``."""
    align = npu_last_dim_elem_align(dtype)
    return int(n) % align == 0


def npu_verify_last_dims(
    *dims: int,
    labels: tuple[str, ...] | None = None,
    dtypes: tuple[torch.dtype, ...] | None = None,
) -> tuple[bool, str | None]:
    """Verifier helper: reject last dims whose byte stride is not 32-byte aligned."""
    labels = labels or tuple(f'dim{i}' for i in range(len(dims)))
    if dtypes is None:
        dtypes = (torch.float16,) * len(dims)
    if len(dtypes) != len(dims):
        raise ValueError(f'expected {len(dims)} dtypes, got {len(dtypes)}')
    bad: list[str] = []
    for label, d, dtype in zip(labels, dims, dtypes, strict=True):
        align = npu_last_dim_elem_align(dtype)
        if int(d) % align != 0:
            bad.append(f'{label}={d} (needs elem align {align} for {dtype})')
    if not bad:
        return True, None
    detail = ', '.join(bad)
    return False, (
        f'triton_ascend requires last-dim byte stride % {NPU_DMA_ALIGN_BYTES} == 0; '
        f'got unsupported {detail} (see fla-org/flash-linear-attention#1250)'
    )


def npu_require_last_dims(
    *dims: int,
    labels: tuple[str, ...] | None = None,
    dtypes: tuple[torch.dtype, ...] | None = None,
) -> None:
    """Raise ``ValueError`` when any last dim is not 32-byte aligned."""
    ok, reason = npu_verify_last_dims(*dims, labels=labels, dtypes=dtypes)
    if not ok:
        raise ValueError(reason)


_NPU_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def npu_verify_last_dim_tensor(
    t: torch.Tensor,
    *,
    label: str = 'dim',
    supported_dtypes: tuple[torch.dtype, ...] | None = _NPU_FLOAT_DTYPES,
) -> tuple[bool, str | None]:
    """Verifier helper for a single tensor's last dim."""
    ok, reason = npu_verify_last_dims(t.shape[-1], labels=(label,), dtypes=(t.dtype,))
    if not ok:
        return ok, reason
    if supported_dtypes is not None and t.dtype not in supported_dtypes:
        return False, f'unsupported dtype for NPU kernels: {t.dtype}'
    return True, None


def npu_verify_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    extra: tuple[torch.Tensor, ...] = (),
    labels: tuple[str, str] = ('K', 'V'),
    supported_dtypes: tuple[torch.dtype, ...] = _NPU_FLOAT_DTYPES,
) -> tuple[bool, str | None]:
    """Verifier helper: 32-byte last-dim stride and float dtypes."""
    ok, reason = npu_verify_last_dims(
        k.shape[-1], v.shape[-1], labels=labels, dtypes=(k.dtype, v.dtype),
    )
    if not ok:
        return ok, reason
    for t in (k, v, *extra):
        if t.dtype not in supported_dtypes:
            return False, f'unsupported dtype for NPU kernels: {t.dtype}'
    return True, None


def npu_leftover_mask(
    *,
    T: int | None = None,
    BT: int | None = None,
    varlen: bool = False,
    K: int | None = None,
    BK: int | None = None,
    V: int | None = None,
    BV: int | None = None,
) -> bool:
    """True if a kernel must zero leftover T lanes (partial last tile / varlen).

    Last-dim leftover-K is cleared by honest blocked-tile shape plus boundary checks.
    ``K`` / ``BK`` / ``V`` / ``BV`` are accepted for call-site compatibility and ignored.
    Gating ``tl.where`` on ``K % BK`` would force leftover copies on D=48/96,
    and would shrink Cube tiles on those 32-byte-aligned shapes.
    """
    del K, BK, V, BV
    if varlen:
        return True
    return T is not None and BT is not None and int(T) % int(BT) != 0
