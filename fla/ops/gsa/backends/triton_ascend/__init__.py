# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for GSA (Gated Slot Attention) ops.

Design notes:
    - Dispatcher methods mirror the names of the upstream functions decorated
      with ``@dispatch('gsa')`` in ``fla/ops/gsa/chunk.py`` and
      ``fla/ops/gsa/fused_recurrent.py``.
    - Each dispatchable method has a companion ``<name>_verifier`` that decides
      whether the NPU backend can handle the call; on rejection the dispatcher
      transparently falls back to the upstream GPU implementation.
    - Verifier limits are conservative on purpose — they will be tightened as
      the kernels land and we measure UB / coreDim envelopes per platform
      (A2 / A3 / 950).
"""

from __future__ import annotations

import functools
import logging
import os
import traceback

import triton

# Force IEEE-precision f32 accumulation for GSA kernels on Ascend NPU. The
# default TF32-like reduced-precision mode on Triton-Ascend 3.2.2 gives
# non-deterministic dg / dv reductions at large T (observed ~14% flake rate
# on ``test_fused_recurrent[B2-T1024-...]``). ``TRITON_F32_DEFAULT=ieee``
# stabilises those reductions to within the test tolerance. Set at import
# so it takes effect before Triton reads the env for kernel compile.
os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')

from fla.ops.backends import BaseBackend
from fla.ops.gsa.backends.triton_ascend.fused_recurrent import _MAX_BK, _MAX_BV

logger = logging.getLogger(__name__)


def _guarded_launch(kernel_fn):
    """Decorator: catch launch errors and emit context (grid, kwargs keys).

    Used on the public NPU wrapper entry points (e.g. ``fused_recurrent_gsa_*_npu``)
    so that 507014/507034 timeout errors and compiler crashes include the
    invocation grid and key kernel-constant values in the message.
    """
    @functools.wraps(kernel_fn)
    def wrapper(*a, **kw):
        try:
            return kernel_fn(*a, **kw)
        except Exception as e:
            logger.error(
                "[FLA GSA NPU] %s raised %s: %s | grid keys=%s | first kw=%s",
                getattr(kernel_fn, '__name__', kernel_fn),
                type(e).__name__,
                e,
                sorted(kw.keys())[:8],
                {k: kw[k] for k in list(kw)[:3] if not hasattr(kw[k], 'shape')},
            )
            logger.debug("[FLA GSA NPU] traceback:\n%s", traceback.format_exc())
            raise
    return wrapper

# Conservative initial limits. Tighten once each kernel is profiled on
# A2 / A3 / 950.
_SUPPORTED_CHUNK_SIZES = (16, 32, 64)
_MAX_HEAD_DIM = 256        # D dimension (K/V in attention)
_MAX_SLOT_DIM = 256        # M dimension (slot/gate)
_MIN_HEAD_DIM = 16
_MIN_SLOT_DIM = 16


def _verify_shapes(q, k, v, s, *args, **kwargs) -> tuple[bool, str | None]:
    """Common shape constraints for GSA forward/backward.

    Checks the K/V head dim and slot dim are within NPU-friendly bounds.
    chunk_size is verified separately by each helper so it can read it from
    the appropriate kwarg.
    """
    try:
        D = int(q.shape[-1])
        M = int(s.shape[-1])
    except (AttributeError, IndexError, TypeError) as e:
        return False, f'GSA Ascend verifier: cannot read shape ({e})'
    if not (_MIN_HEAD_DIM <= D <= _MAX_HEAD_DIM):
        return False, f'GSA Ascend only supports D in [{_MIN_HEAD_DIM}, {_MAX_HEAD_DIM}], got D={D}'
    if not (_MIN_SLOT_DIM <= M <= _MAX_SLOT_DIM):
        return False, f'GSA Ascend only supports M in [{_MIN_SLOT_DIM}, {_MAX_SLOT_DIM}], got M={M}'
    return True, None


def _verify_chunk_size(chunk_size) -> tuple[bool, str | None]:
    if chunk_size is None:
        return True, None
    if chunk_size not in _SUPPORTED_CHUNK_SIZES:
        return False, (
            f'GSA Ascend only supports chunk_size in {_SUPPORTED_CHUNK_SIZES}, '
            f'got chunk_size={chunk_size}'
        )
    return True, None


class TritonAscendGSABackend(BaseBackend):
    """Ascend NPU backend for GSA chunk + fused_recurrent paths."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0  # lowest number = highest priority

    def __repr__(self) -> str:
        from fla.utils import IS_NPU
        return (
            f"<TritonAscendGSABackend active={IS_NPU} "
            f"chunk_size={_SUPPORTED_CHUNK_SIZES} "
            f"D∈[{_MIN_HEAD_DIM},{_MAX_HEAD_DIM}] "
            f"M∈[{_MIN_SLOT_DIM},{_MAX_SLOT_DIM}]>"
        )

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    # ------------------------------------------------------------------
    # Chunk path — forward
    # ------------------------------------------------------------------
    def chunk_gsa_fwd_k_verifier(self, q, k, v, g, *args, **kwargs):
        # `v` here is the slot tensor (upstream convention: chunk_gsa_fwd_k is
        # invoked from `chunk_gsa_fwd` with `v=s`); it carries the slot dim M.
        ok, reason = _verify_shapes(q, k, v, g, *args, **kwargs)
        if not ok:
            return ok, reason
        return _verify_chunk_size(kwargs.get('chunk_size'))

    def chunk_gsa_fwd_k(self, *args, **kwargs):
        from fla.ops.gsa.backends.triton_ascend.chunk import chunk_gsa_fwd_k_npu
        return chunk_gsa_fwd_k_npu(*args, **kwargs)

    def chunk_gsa_fwd_v_verifier(self, q, k, v, g, *args, **kwargs):
        ok, reason = _verify_shapes(q, k, v, g, *args, **kwargs)
        if not ok:
            return ok, reason
        return _verify_chunk_size(kwargs.get('chunk_size'))

    def chunk_gsa_fwd_v(self, *args, **kwargs):
        from fla.ops.gsa.backends.triton_ascend.chunk import chunk_gsa_fwd_v_npu
        return chunk_gsa_fwd_v_npu(*args, **kwargs)

    # ------------------------------------------------------------------
    # Chunk path — backward
    # ------------------------------------------------------------------
    def chunk_gsa_bwd_k_verifier(self, q, k, v, g, *args, **kwargs):
        # `v` here is again the slot tensor (M-axis), see chunk_gsa_fwd_k.
        ok, reason = _verify_shapes(q, k, v, g, *args, **kwargs)
        if not ok:
            return ok, reason
        return _verify_chunk_size(kwargs.get('chunk_size'))

    def chunk_gsa_bwd_k(self, *args, **kwargs):
        from fla.ops.gsa.backends.triton_ascend.chunk import chunk_gsa_bwd_k_npu
        return chunk_gsa_bwd_k_npu(*args, **kwargs)

    def chunk_gsa_bwd_v_verifier(self, q, k, v, g, *args, **kwargs):
        ok, reason = _verify_shapes(q, k, v, g, *args, **kwargs)
        if not ok:
            return ok, reason
        return _verify_chunk_size(kwargs.get('chunk_size'))

    def chunk_gsa_bwd_v(self, *args, **kwargs):
        from fla.ops.gsa.backends.triton_ascend.chunk import chunk_gsa_bwd_v_npu
        return chunk_gsa_bwd_v_npu(*args, **kwargs)

    # ------------------------------------------------------------------
    # Fused recurrent path
    # ------------------------------------------------------------------
    def fused_recurrent_gsa_inference_verifier(self, q, k, v, s, g, *args, **kwargs):
        ok, reason = _verify_shapes(q, k, v, s, *args, **kwargs)
        if not ok:
            return ok, reason
        # The inference kernel uses a single K-tile (BK >= K) and a single
        # V-tile (BV >= V); refuse when the dims would spill so the dispatcher
        # transparently falls back to the GPU upstream kernel.
        K = int(k.shape[-1])
        V = int(v.shape[-1])
        BK = min(triton.next_power_of_2(K), 64)
        BV = min(triton.next_power_of_2(V), 256)
        if BK < K:
            return False, f'GSA Ascend inference requires K <= {_MAX_BK}, got K={K}'
        if BV < V:
            return False, f'GSA Ascend inference requires V <= {_MAX_BV}, got V={V}'
        return True, None

    def fused_recurrent_gsa_inference(self, *args, **kwargs):
        from fla.ops.gsa.backends.triton_ascend.fused_recurrent import (
            fused_recurrent_gsa_inference_npu,
        )
        return fused_recurrent_gsa_inference_npu(*args, **kwargs)

    def fused_recurrent_gsa_fwd_verifier(self, q, k, v, s, g, *args, **kwargs):
        return _verify_shapes(q, k, v, s, *args, **kwargs)

    def fused_recurrent_gsa_fwd(self, *args, **kwargs):
        from fla.ops.gsa.backends.triton_ascend.fused_recurrent import (
            fused_recurrent_gsa_fwd_npu,
        )
        return fused_recurrent_gsa_fwd_npu(*args, **kwargs)

    def fused_recurrent_gsa_bwd_verifier(self, q, k, v, s, g, *args, **kwargs):
        return _verify_shapes(q, k, v, s, *args, **kwargs)

    def fused_recurrent_gsa_bwd(self, *args, **kwargs):
        from fla.ops.gsa.backends.triton_ascend.fused_recurrent import (
            fused_recurrent_gsa_bwd_npu,
        )
        return fused_recurrent_gsa_bwd_npu(*args, **kwargs)
