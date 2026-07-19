"""Triton backend helpers for delta-rule ops."""

from fla.ops.delta_rule.backends.triton.chunk_bwd import (
    chunk_delta_rule_wy_dqkw_fused_triton,
)

__all__ = ["chunk_delta_rule_wy_dqkw_fused_triton"]

