# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Static buffer management for platform graph capture (CUDA graph, NPU graph).

Buffers are allocated on first use, kept for the process lifetime, and shared by all
graphs with the same (name, shape, dtype, device) key; interleaved replay of graphs
with identical keys is not supported.
"""

import torch

_BUFFERS: dict[tuple, torch.Tensor] = {}


def get_static_buffer(
    name: str,
    shape: tuple,
    dtype: torch.dtype,
    device: torch.device | str,
    zero: bool = False,
) -> torch.Tensor:
    """Return the persistent buffer for the given key, allocated on first use and reused after.

    With ``zero=True`` the buffer is zeroed on every call, not just at allocation.
    Ascend graph replay requires padding rows to be physically zero: tl.dot padding
    can carry UB residue, and NaN x 0 = NaN defeats zero-weight masks. Called inside
    the capture region, the zero-fill is recorded into the graph and re-executed on
    every replay.
    """
    key = (name, tuple(shape), dtype, str(device))
    buf = _BUFFERS.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _BUFFERS[key] = buf
    return buf.zero_() if zero else buf
