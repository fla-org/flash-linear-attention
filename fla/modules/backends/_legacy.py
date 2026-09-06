# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compatibility registry for the deprecated modules-wide dispatch key."""

from fla.backends import BackendRegistry
from fla.modules.backends.activations import TritonAscendActivationsBackend
from fla.modules.backends.causal_conv1d import TritonAscendCausalConv1dBackend
from fla.modules.backends.fused_cross_entropy import TritonAscendFusedCrossEntropyBackend
from fla.modules.backends.fused_kl_div import TritonAscendFusedKLDivBackend
from fla.modules.backends.fused_linear_cross_entropy import TritonAscendFusedLinearCrossEntropyBackend
from fla.modules.backends.fused_norm_gate import TritonAscendFusedNormGateBackend
from fla.modules.backends.grpo import TritonAscendGRPOBackend
from fla.modules.backends.l2norm import TritonAscendL2NormBackend
from fla.modules.backends.layernorm import TritonAscendLayerNormBackend
from fla.modules.backends.rotary import TritonAscendRotaryBackend


class TritonAscendBackend(
    TritonAscendActivationsBackend,
    TritonAscendCausalConv1dBackend,
    TritonAscendFusedCrossEntropyBackend,
    TritonAscendFusedKLDivBackend,
    TritonAscendFusedLinearCrossEntropyBackend,
    TritonAscendFusedNormGateBackend,
    TritonAscendGRPOBackend,
    TritonAscendL2NormBackend,
    TritonAscendLayerNormBackend,
    TritonAscendRotaryBackend,
):
    pass


class _ModulesRegistry(BackendRegistry):
    def register(self, backend):
        super().register(backend)
        for operation, registry in self._registries.items():
            if operation.startswith('modules.'):
                registry.register(backend)


modules_registry = _ModulesRegistry("modules")
BackendRegistry.register(modules_registry, TritonAscendBackend())
