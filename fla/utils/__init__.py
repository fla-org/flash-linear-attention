# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import sys

from ._compat import (  # noqa: F401
    SUPPORTS_AUTOTUNE_CACHE,
    TRITON_ABOVE_3_4_0,
    TRITON_ABOVE_3_5_1,
    TRITON_ABOVE_3_7_1,
    autotune_cache_kwargs,
    find_spec_cached,
)
from ._config import (  # noqa: F401
    FLA_CACHE_RESULTS,
    FLA_CI_ENV,
    FLA_DISABLE_TENSOR_CACHE,
    FLA_TENSOR_CACHE_SIZE,
)
from ._decorators import (  # noqa: F401
    Action,
    checkpoint,
    contiguous,
    deprecate_kwarg,
    input_guard,
    require_version,
    tensor_cache,
)
from ._device import (  # noqa: F401
    IS_AMD,
    IS_ARM,
    IS_GATHER_SUPPORTED,
    IS_INTEL,
    IS_INTEL_ALCHEMIST,
    IS_NPU,
    IS_NVIDIA,
    IS_NVIDIA_BLACKWELL,
    IS_NVIDIA_HOPPER,
    IS_TF32_SUPPORTED,
    IS_TMA_SUPPORTED,
    Backend,
    autocast_custom_bwd,
    autocast_custom_fwd,
    check_environments,
    check_pytorch_version,
    check_shared_mem,
    custom_device_ctx,
    device,
    device_name,
    device_platform,
    device_torch_lib,
    get_all_max_shared_mem,
    get_available_device,
    get_multiprocessor_count,
    map_triton_backend_to_torch_device,
)
from ._testing import assert_close, get_abs_err, get_err_ratio  # noqa: F401


def _register_aliases():
    current_module = sys.modules[__name__]
    for key in (
        'IS_AMD',
        'IS_ARM',
        'IS_INTEL',
        'IS_INTEL_ALCHEMIST',
        'IS_NVIDIA',
        'IS_NPU',
        'IS_NVIDIA_BLACKWELL',
        'IS_NVIDIA_HOPPER',
        'IS_TF32_SUPPORTED',
        'IS_GATHER_SUPPORTED',
        'IS_TMA_SUPPORTED',
    ):
        if hasattr(current_module, key):
            setattr(current_module, key.lower(), getattr(current_module, key))


_register_aliases()

del _register_aliases
