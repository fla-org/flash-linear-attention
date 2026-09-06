# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Deprecated import path for the causal_conv1d Triton-Ascend implementation."""

from importlib import import_module

_RENAMED = {
    '_npu_bwd_tile_config': '_bwd_tile_config',
    '_npu_chunk_size': '_chunk_size',
    '_npu_max_axis_chunks': '_max_axis_chunks',
    '_npu_tile_config': '_tile_config',
    'causal_conv1d_bwd_npu': 'causal_conv1d_bwd',
    'causal_conv1d_fwd_npu': 'causal_conv1d_fwd',
    'causal_conv1d_update_npu': 'causal_conv1d_update',
    'causal_conv1d_update_states_npu': 'causal_conv1d_update_states',
    'compute_dh0_npu': 'compute_dh0',
}

_PUBLIC_NAMES = (
    'causal_conv1d_bwd_coregrid_kernel',
    'causal_conv1d_bwd_dwdb_kernel',
    'causal_conv1d_bwd_dx_kernel',
    'causal_conv1d_bwd_npu',
    'causal_conv1d_bwd_seq_kernel',
    'causal_conv1d_fwd_coregrid_kernel',
    'causal_conv1d_fwd_kernel',
    'causal_conv1d_fwd_kernel_scalar',
    'causal_conv1d_fwd_npu',
    'causal_conv1d_states_fwd_kernel',
    'causal_conv1d_update_kernel',
    'causal_conv1d_update_npu',
    'causal_conv1d_update_states_npu',
    'compute_dh0_kernel',
    'compute_dh0_npu',
)


def __getattr__(name):
    if name == "__all__":
        return _PUBLIC_NAMES
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("fla.modules.backends.causal_conv1d.triton_ascend")
    return getattr(module, _RENAMED.get(name, name))
