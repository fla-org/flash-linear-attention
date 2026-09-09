# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Intra-card CP backend for shared delta rule operations.

Accelerates prefill by splitting long sequences into sub-sequences
and processing them in parallel across SMs.

Only active under torch.inference_mode() with varlen (cu_seqlens != None).
"""

from __future__ import annotations

import os
import warnings

import torch

from fla.ops.backends import BaseBackend
from fla.utils import IS_TF32_SUPPORTED

# Maximum number of sub-sequences per original sequence
# Limits merge chain depth to control precision loss
MAX_SUBSEQS = int(os.environ.get('FLA_INTRACARD_MAX_SPLITS', 32))

# use tf32x3 for the affine-chain dots in the pre-scan/merge kernels (NVIDIA only)
USE_TF32X3_AFFINE_CHAIN = os.environ.get('FLA_INTRACARD_TF32X3', '0') == '1'

if USE_TF32X3_AFFINE_CHAIN and not IS_TF32_SUPPORTED:
    warnings.warn(
        "tf32x3 affine chain requires an NVIDIA GPU with compute capability >= 8.0; falling back to ieee precision",
        stacklevel=2,
    )


class IntraCardCPBackend(BaseBackend):
    """Intra-card context parallel backend for chunk_gated_delta_rule_fwd_h."""

    backend_type = "intracard_cp"
    package_name = None  # No external package needed
    env_var = "FLA_INTRACARD_CP"
    default_enable = False

    @classmethod
    def is_available(cls) -> bool:
        return True

    def chunk_gated_delta_rule_fwd_h_verifier(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        chunk_size: int = 64,
        save_new_value: bool = True,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[bool, str | None]:
        """Check if intracard CP should handle this call."""
        # Only in inference mode
        if not torch.is_inference_mode_enabled():
            return False, "Not in inference mode"

        # Only for varlen
        if cu_seqlens is None:
            return False, "cu_seqlens is None"

        return True, None

    def chunk_gated_delta_rule_fwd_h(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        chunk_size: int = 64,
        save_new_value: bool = True,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Intra-card CP implementation of chunk_gated_delta_rule_fwd_h."""
        from fla.ops.common.intracard_cp import intracard_fwd_h

        return intracard_fwd_h(
            k=k, w=w, u=u, g=g, gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            max_splits=MAX_SUBSEQS,
            state_v_first=state_v_first,
            use_tf32x3_affine_chain=USE_TF32X3_AFFINE_CHAIN,
        )
