from fla.ops.backends import BackendRegistry
from .triton_ascend import TritonAscendOpsBackend

registry = BackendRegistry("gated_delta_rule")
registry.register(TritonAscendOpsBackend())
