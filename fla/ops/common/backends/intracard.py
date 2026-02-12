"""Intra-card CP backend for shared delta rule operations.

Accelerates prefill by splitting long sequences into sub-sequences
and processing them in parallel across SMs.

Only active under torch.inference_mode() with varlen (cu_seqlens != None).
"""

from __future__ import annotations

import os

import torch

from fla.ops.backends import BaseBackend

# Maximum number of sub-sequences per original sequence
# Limits merge chain depth to control precision loss
MAX_SUBSEQS = int(os.environ.get('FLA_INTRACARD_MAX_SPLITS', 32))


class IntraCardCPBackend(BaseBackend):
    """Intra-card context parallel backend for chunk_gated_delta_rule_fwd_h."""

    backend_type = "intracard_cp"
    package_name = None  # No external package needed
    env_var = "FLA_INTRACARD_CP"

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
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        use_exp2: bool = False,
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
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        use_exp2: bool = False,
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
            use_exp2=use_exp2,
            max_splits=MAX_SUBSEQS,
        )
