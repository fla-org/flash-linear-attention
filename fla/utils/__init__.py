# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.utils.decorators import (
    Action,
    checkpoint,
    contiguous,
    deprecate_kwarg,
    input_guard,
    require_version,
    tensor_cache,
)
from fla.utils.env import (
    FLA_CACHE_RESULTS,
    FLA_CI_ENV,
    FLA_DISABLE_TENSOR_CACHE,
    FLA_TENSOR_CACHE_SIZE,
    SUPPORTS_AUTOTUNE_CACHE,
    TRITON_ABOVE_3_4_0,
    TRITON_ABOVE_3_5_1,
    TRITON_ABOVE_3_6_0,
    TRITON_ABOVE_3_7_1,
    TRITON_ABOVE_3_8_0,
    autotune_cache_kwargs,
    check_environments,
    check_pytorch_version,
    find_spec_cached,
    has_usable_nvcc,
)
from fla.utils.hardware import (
    IS_AMD,
    IS_AMD_TMA_ARCH,
    IS_ARM,
    IS_GATHER_SUPPORTED,
    IS_INTEL,
    IS_INTEL_ALCHEMIST,
    IS_NPU,
    IS_NVIDIA,
    IS_NVIDIA_BLACKWELL,
    IS_NVIDIA_HOPPER,
    IS_NVIDIA_SM100,
    IS_NVIDIA_SM120,
    IS_TF32_SUPPORTED,
    IS_TMA_SUPPORTED,
    Backend,
    ascend_compile_kwargs,
    autocast_custom_bwd,
    autocast_custom_fwd,
    check_shared_mem,
    custom_device_ctx,
    device,
    device_name,
    device_platform,
    device_torch_lib,
    get_all_max_shared_mem,
    get_available_device,
    get_device_arch,
    get_device_capability,
    get_device_smem_optin,
    get_multiprocessor_count,
    map_triton_backend_to_torch_device,
    pytorch_matmul_config,
)
from fla.utils.testing import assert_close, get_abs_err, get_err_ratio

is_amd = IS_AMD
is_amd_tma_arch = IS_AMD_TMA_ARCH
is_arm = IS_ARM
is_intel = IS_INTEL
is_intel_alchemist = IS_INTEL_ALCHEMIST
is_nvidia = IS_NVIDIA
is_npu = IS_NPU
is_nvidia_blackwell = IS_NVIDIA_BLACKWELL
is_nvidia_hopper = IS_NVIDIA_HOPPER
is_nvidia_sm100 = IS_NVIDIA_SM100
is_nvidia_sm120 = IS_NVIDIA_SM120
is_tf32_supported = IS_TF32_SUPPORTED
is_gather_supported = IS_GATHER_SUPPORTED
is_tma_supported = IS_TMA_SUPPORTED
