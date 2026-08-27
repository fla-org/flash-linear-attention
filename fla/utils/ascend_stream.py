# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Stream-lifetime helpers for asynchronous Triton-Ascend launches."""

from __future__ import annotations

from collections import deque
from threading import Lock

import torch

_PENDING_RELEASES: deque[tuple[object, tuple[torch.Tensor, ...]]] = deque()
_PENDING_RELEASES_LOCK = Lock()


def defer_npu_tensor_release(*tensors: torch.Tensor | None) -> None:
    """Retain temporary tensors until work queued on their NPU stream completes.

    ``Tensor.record_stream`` alone is insufficient for raw Triton-Ascend
    launches in torch_npu's asynchronous execution mode: Python can release a
    temporary before the launch has registered its raw pointer with the caching
    allocator. Holding a reference behind a stream event prevents reuse without
    introducing a device synchronization.
    """
    live_tensors = tuple(tensor for tensor in tensors if tensor is not None)
    if not live_tensors:
        return
    stream = torch.npu.current_stream(live_tensors[0].device)
    for tensor in live_tensors:
        tensor.record_stream(stream)
    event = torch.npu.Event()
    event.record(stream)

    with _PENDING_RELEASES_LOCK:
        if _PENDING_RELEASES:
            pending = deque(
                (pending_event, pending_tensors)
                for pending_event, pending_tensors in _PENDING_RELEASES
                if not pending_event.query()
            )
            _PENDING_RELEASES.clear()
            _PENDING_RELEASES.extend(pending)
        _PENDING_RELEASES.append((event, live_tensors))


__all__ = ['defer_npu_tensor_release']
