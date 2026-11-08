# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'PHBA input validation, structured errors and gradient alignment.'

from __future__ import annotations

from typing import Any

import torch

CLASS_A = "A"


CLASS_C = "C"


REASON_CLASSES: dict[str, str] = {
    "device_mismatch": CLASS_A,
    "dtype_mismatch": CLASS_A,
    "non_contiguous_input": CLASS_A,
    "invalid_shape": CLASS_A,
    "invalid_cu_seqlens": CLASS_A,
    "int32_overflow": CLASS_A,
    "routing_unsupported": CLASS_A,
    "fa_layout_probe_failed": CLASS_A,
    "fa_device_unsupported": CLASS_A,
    "invalid_config": CLASS_A,
    "prefix_kernel_unsupported": CLASS_C,
}


class PHBAError(RuntimeError):
    """PHBA runtime error carrying a registered reason and optional diagnostic details."""

    def __init__(self, reason: str, message: str = "", **details: Any) -> None:
        if reason not in REASON_CLASSES:
            raise ValueError(f"Unregistered PHBA fallback_reason: {reason!r}")
        self.fallback_reason = reason
        self.fallback_class = REASON_CLASSES[reason]
        self.details = details
        text = f"[{reason}] {message}" if message else f"[{reason}]"
        super().__init__(text)


def fail(reason: str, message: str = "", **details: Any) -> PHBAError:
    """Construct a structured PHBA runtime error."""
    return PHBAError(reason, message, **details)


ALIGNMENT_BYTES = 16


def prepare_grad_output(grad_out: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Validate the output gradient and copy unaligned views into fresh storage.

    A contiguous view can still be unaligned; contiguity alone does not satisfy the alignment contract.
    """
    if grad_out.dtype != out.dtype:
        raise fail("dtype_mismatch", f"dO dtype {grad_out.dtype} does not match output dtype {out.dtype}")
    if tuple(grad_out.shape) != tuple(out.shape):
        raise fail("invalid_shape", f"dO shape {tuple(grad_out.shape)} does not match output shape {tuple(out.shape)}")
    if grad_out.stride(-1) == 1 and grad_out.data_ptr() % ALIGNMENT_BYTES == 0:
        return grad_out
    fresh = torch.empty(grad_out.shape, dtype=grad_out.dtype, device=grad_out.device)
    fresh.copy_(grad_out)
    return fresh
