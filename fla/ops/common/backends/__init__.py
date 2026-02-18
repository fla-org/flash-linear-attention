"""Common backends for shared operations like chunk_gated_delta_rule_fwd_h."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.common.backends.intracard import IntraCardCPBackend

common_registry = BackendRegistry("common")


common_registry.register(IntraCardCPBackend())


__all__ = ['common_registry', 'dispatch']
