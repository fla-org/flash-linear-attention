# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Compile-only recurrence address checks; these never execute oversized tensors."""

import json
import re
from importlib import import_module

import pytest
import torch
from triton.runtime.jit import JITFunction

from fla.utils import IS_NVIDIA, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA compiled address checks require NVIDIA CUDA")

KERNELS = [
    pytest.param("nongated_fwd", "fused_recurrent_swila_fwd_kernel", id="plain-forward"),
    pytest.param("nongated_bwd", "fused_recurrent_swila_bwd_kernel", id="plain-backward"),
    pytest.param("gated_fwd", "fused_recurrent_gated_swila_fwd_kernel", id="gated-forward"),
    pytest.param("gated_bwd", "fused_recurrent_gated_swila_bwd_kernel", id="gated-backward"),
    pytest.param("gated_split_bwd", "_fused_recurrent_gated_swila_dense_d64e4_bwd_reverse_kernel", id="gated-split-reverse"),
    pytest.param("gated_split_bwd", "_fused_recurrent_gated_swila_dense_d64e4_bwd_replay_kernel", id="gated-split-replay"),
    pytest.param("gated_split_bwd", "_reverse_cumsum_log_decay_kernel", id="gated-decay-scan"),
    pytest.param("temporal_fwd", "fused_recurrent_swila_temporal_fwd_kernel", id="temporal-forward"),
    pytest.param("temporal_bwd_state", "fused_recurrent_swila_temporal_bwd_kernel", id="temporal-state-backward"),
    pytest.param("temporal_split_bwd", "fused_recurrent_swila_temporal_bwd_q_temporal_kernel", id="temporal-query-backward"),
    pytest.param("temporal_split_bwd", "fused_recurrent_swila_temporal_bwd_replay_kernel", id="temporal-replay"),
    pytest.param("temporal_bwd", "_reverse_cumsum_log_decay_kernel", id="temporal-decay-scan"),
]


def address_arguments(kernel, *, dense_split):
    metadata = dict(
        T=2**32 + 65, B=2, H=2**20 + 3, scale=.7,
        D_k=64 if dense_split else 5, D_v=64 if dense_split else 3, E=4 if dense_split else 3,
        BK=64 if dense_split else 8, BV=16 if dense_split else 8, BE=4, NK=1,
        USE_INITIAL_STATE=True, STORE_FINAL_STATE=True, STORE_INTERMEDIATES=True,
        STORE_R=True, STORE_U=True, STORE_PI_BAR=True, STORE_RHO=True, STORE_QS=True,
        USE_INITIAL_TEMPORAL_STATE=True, STORE_TEMPORAL_STATE_GRADIENT=True,
        USE_FINAL_STATE_GRADIENT=True, USE_FINAL_R_GRADIENT=True, USE_FINAL_RHO_GRADIENT=True,
        USE_AUX_LOSS_GRAD=True, USE_U_AUX_GRAD=True, USE_PI_BAR_AUX_GRAD=True,
        USE_GATE=True, IDENTITY_TRANSITIONS=True, USE_LOG_DECAY=True, IS_VARLEN=True,
        DEFER_READOUT=False, DEBUG_INTERMEDIATES=False, SKIP_REPLAY=False,
        SPLIT_Q_TEMPORAL=False, REUSE_DK_COEF=False, RESOURCE_SAFE=False, PRODUCTION_FAST=False,
    )
    buffer = torch.empty(1, device=device, dtype=torch.float32)
    boundaries = torch.empty(1, device=device, dtype=torch.int64)
    arguments = {}
    for index, name in enumerate(kernel.arg_names):
        if name in metadata:
            arguments[name] = metadata[name]
        else:
            assert index not in kernel.constexprs, f"missing explicit compile metadata: {name}"
            arguments[name] = boundaries if name == "cu_seqlens" else buffer
    return arguments


def check_recurrence_address_ir(ir):
    assert not re.search(r"(?:\bf64\b|xf64\b)", ir), "address widening must not introduce FP64 arithmetic"
    additions = [line for line in ir.splitlines() if "tt.addptr" in line]
    assert additions, "compiled kernel has no address operations"
    constants = {
        name: int(value) for name, value in re.findall(
            r"^\s*(%\w+)\s*= arith\.constant (-?\d+) : i32\b", ir, flags=re.MULTILINE,
        )
    }
    constants.update({
        name: int(value) for name, value in re.findall(
            r"^\s*(%\w+)\s*= arith\.constant dense<(-?\d+)> : tensor<[^>]*xi32>", ir, flags=re.MULTILINE,
        )
    })
    for name, scalar in re.findall(
        r"^\s*(%\w+)\s*= tt\.splat (%\w+) : i32 -> tensor<[^>]*xi32>", ir, flags=re.MULTILINE,
    ):
        if scalar in constants:
            constants[name] = constants[scalar]
    wide = 0
    for line in additions:
        if re.search(r"(?:\bi64\b|xi64\b)", line):
            wide += 1
            continue
        # The compiler may narrow fixed channels or loop strides; these constants cannot overflow.
        # Splatting a proven constant is safe too; dynamic i32 offsets are not.
        offset = re.search(r"tt\.addptr %\w+, (%\w+)\s*:", line)
        assert re.search(r"(?:\bi32\b|xi32\b)", line), line
        assert offset is not None and offset.group(1) in constants, line
        assert -(2**31) <= constants[offset.group(1)] < 2**31, line
    assert wide > 0
    assert re.search(r"arith\.(?:muli|shli).*?(?:\bi64\b|xi64\b)", ir), "address products must be int64"
    return {"pointer_additions": len(additions), "wide_pointer_additions": wide}


@pytest.mark.parametrize(("module_name", "kernel_name"), KERNELS)
def test_swila_recurrence_large_address_ir(module_name, kernel_name, tmp_path, record_property):
    kernel = getattr(import_module(f"fla.ops.swila.{module_name}"), kernel_name)
    while not isinstance(kernel, JITFunction):
        kernel = kernel.fn
    dense_split = module_name == "gated_split_bwd" and "dense_d64e4" in kernel_name
    arguments = address_arguments(kernel, dense_split=dense_split)
    stages = 1 if dense_split and "replay" in kernel_name else 2
    # warmup compiles the actual body without launching it or running autotune hooks.
    compiled = kernel.warmup(**arguments, grid=(2, 2, 2), num_warps=1, num_stages=stages)
    ir = compiled.asm["ttir"]
    (tmp_path / f"{module_name}-{kernel_name}.ttir").write_text(ir)
    assert compiled.src.signature["T"] == "i64"
    summary = check_recurrence_address_ir(ir)
    record_property("compiled_addresses", json.dumps({
        "module": module_name, "kernel": kernel_name, "hash": compiled.hash,
        "execution": "compile-only", "pointer_elements": 1,
        "metadata": {name: value for name, value in arguments.items() if not isinstance(value, torch.Tensor)},
        **summary,
    }, sort_keys=True))
