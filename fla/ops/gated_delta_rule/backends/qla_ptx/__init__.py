# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
"""B300 PTX forward backend for preprocessed gated delta rule inputs."""

import inspect
import json
import os
import shlex
from pathlib import Path
from shutil import which

import torch

from fla.backends import BaseBackend, register
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from fla.utils import custom_device_ctx, get_device_capability, has_usable_nvcc

WORKLOADS = json.loads((Path(__file__).parent / "workloads.json").read_text())


@register(chunk_gated_delta_rule)
class QLAPTXBackend(BaseBackend):
    backend_type = "qla_ptx"
    package_name = "tvm_ffi"
    priority = 2

    @classmethod
    def is_available(cls):
        if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
            return False
        if not (super().is_available() and has_usable_nvcc()):
            return False
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or which(compiler[0]) is None or which("ninja") is None:
            return False
        try:
            from torch.utils.cpp_extension import CUDA_HOME
            from tvm_ffi import cpp
        except ImportError:
            return False
        return CUDA_HOME is not None and "embed_cubin" in inspect.signature(cpp.load_inline).parameters

    def chunk_gated_delta_rule_verifier(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        state_v_first=False,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        cp_context=None,
        return_intermediate_states=False,
        **kwargs,
    ):
        if q.device.type != "cuda" or get_device_capability(q.device.index) != (10, 3):
            return False, "QLA PTX requires NVIDIA SM103"
        if (
            use_qk_l2norm_in_kernel
            or use_beta_sigmoid_in_kernel
            or allow_neg_eigval
            or state_v_first
            or cp_context is not None
            or return_intermediate_states
            or chunk_indices is not None
            or kwargs.get("use_gate_in_kernel", False)
            or kwargs.get("chunk_size", 64) != 64
            or kwargs.get("A_log") is not None
            or kwargs.get("dt_bias") is not None
            or set(kwargs) - {"use_gate_in_kernel", "chunk_size", "A_log", "dt_bias"}
        ):
            return False, "QLA PTX requires preprocessed inputs and K-first states"
        tensors = (q, k, v, g, beta, initial_state, cu_seqlens)
        if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
            return False, "QLA PTX implements forward inference only"
        if any(t is not None and (t.device != q.device or not t.is_contiguous()) for t in tensors):
            return False, "QLA PTX requires contiguous tensors on the same device"
        if not (q.dtype == k.dtype == v.dtype == torch.bfloat16 and g.dtype == beta.dtype == torch.float32):
            return False, "QLA PTX requires BF16 q/k/v and FP32 g/beta"
        if q.ndim != 4 or v.ndim != 4 or k.shape != q.shape:
            return False, "QLA PTX received incompatible q/k/v shapes"
        b, t, hq, dk = q.shape
        hv, dv = v.shape[2:]
        if v.shape[:2] != (b, t) or g.shape != (b, t, hv) or beta.shape != g.shape:
            return False, "QLA PTX received incompatible gate or value shapes"
        matching = [w for w in WORKLOADS if (b, t, hq, hv, dk, dv) == tuple(w[n] for n in ("B", "T", "Hq", "Hv", "Dk", "Dv"))]
        matching = [w for w in matching if ("cu" in w) == (cu_seqlens is not None)]
        if not matching:
            return False, "QLA PTX specialization is not available"
        workload = matching[0]
        if cu_seqlens is not None:
            if cu_seqlens.dtype != torch.int64 or cu_seqlens.shape != (len(workload["cu"]),):
                return False, "QLA PTX requires the exported sequence count and int64 offsets"
            if cu_seqlens_cpu is None or cu_seqlens_cpu.device.type != "cpu":
                return False, "QLA PTX requires CPU sequence metadata to verify packed specializations"
            if cu_seqlens_cpu.tolist() != workload["cu"]:
                return False, "QLA PTX packed sequence boundaries were not exported"
        n = len(workload["cu"]) - 1 if cu_seqlens is not None else b
        if initial_state is not None and (initial_state.shape != (n, hv, dk, dv) or initial_state.dtype != torch.float32):
            return False, "QLA PTX requires FP32 K-first initial states"
        return True, None

    def chunk_gated_delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        state_v_first=False,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        cp_context=None,
        return_intermediate_states=False,
        **kwargs,
    ):
        from fla.ops.gated_delta_rule.backends.qla_ptx.kernel import run

        with custom_device_ctx(q.device.index):
            output, state = run(q, k, v, g, beta, q.shape[-1] ** -0.5 if scale is None else scale, initial_state, cu_seqlens)
        return output, state if output_final_state else None
