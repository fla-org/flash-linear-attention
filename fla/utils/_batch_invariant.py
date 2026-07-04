# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Batch-invariant execution mode.

When this mode is enabled, participating kernels guarantee that the output for
a given token is **bitwise identical** regardless of

- how many other sequences are present in the batch, and
- how the sequence is split across calls, e.g. a single full-sequence call vs.
  a prefill call followed by token-by-token decode calls with the fp32
  recurrent state carried unmodified between them.

Participating ops achieve this by pinning every call to a single compiled
kernel specialization (same ``tl.constexpr`` flags, launch config, and block
sizes for prefill, decode, fixed-length, and varlen calls alike) and by
keeping the recurrent state handoff in fp32, which matches the in-kernel
accumulator and makes the store/reload at call boundaries lossless.

This is a prerequisite for reproducible inference and for exact
training/inference equivalence checks. It comes at a small performance cost
(e.g. always materializing the fp32 state buffers), hence it is opt-in.

The mode can be enabled process-wide via the ``FLA_BATCH_INVARIANT``
environment variable (``1`` / ``true`` / ``yes``, read once at import time),
or at runtime via :func:`set_batch_invariant_mode` or the
:func:`batch_invariant_mode` context manager. The runtime toggles are backed
by a :class:`contextvars.ContextVar` and therefore only affect the current
thread (or async task), so a multi-threaded server can serve invariant and
regular requests concurrently::

    >>> from fla.utils import batch_invariant_mode
    >>> with batch_invariant_mode():
    ...     o, state = fused_recurrent_gated_delta_rule(q, k, v, g, beta, ...)

Currently participating ops:

- ``fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule``
  (splits allowed at any token position)
- ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``
  (splits allowed at chunk-size boundaries; the fp32 state handed from a
  chunked prefill call to recurrent decode calls is lossless, so the
  chunk-prefill + recurrent-decode serving pipeline is deterministic and
  batch-invariant end to end)

Under the mode, autotuned kernels launch a fixed, deterministically chosen
config instead of a timing-based winner, so the compiled binary -- and with it
the in-kernel reduction order -- cannot vary between runs or shapes.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar

_batch_invariant_mode: ContextVar[bool] = ContextVar(
    'fla_batch_invariant_mode',
    default=os.getenv('FLA_BATCH_INVARIANT', '0').lower() in ('1', 'true', 'yes'),
)


def is_batch_invariant_mode_enabled() -> bool:
    """Return whether the batch-invariant mode is enabled in the current context."""
    return _batch_invariant_mode.get()


def set_batch_invariant_mode(enabled: bool = True) -> None:
    """Enable or disable the batch-invariant mode for the current context.

    The setting is scoped to the current thread (or async task). To enable the
    mode process-wide, set ``FLA_BATCH_INVARIANT=1`` before importing ``fla``.
    """
    _batch_invariant_mode.set(enabled)


@contextmanager
def batch_invariant_mode(enabled: bool = True):
    """Context manager that temporarily enables (or disables) the batch-invariant mode."""
    token = _batch_invariant_mode.set(enabled)
    try:
        yield
    finally:
        _batch_invariant_mode.reset(token)
