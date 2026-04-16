# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Tests for int32 overflow in Triton kernel pointer arithmetic.

When B*T*H*K > INT32_MAX (~2.15B), tl.program_id() (int32) multiplied by
large strides can overflow, causing illegal memory accesses or wrong results.

These tests use B=4096, T=576, H=8, K=128, V=128 which gives
B*T*H*K = 2.4B > INT32_MAX. They pass with int64 index casts and
crash (illegal memory access) without them.

Requires ~20GB GPU memory. Run with CUDA_LAUNCH_BLOCKING=1 for
immediate error reporting.
"""

import pytest
import torch
import torch.nn.functional as F

from fla.utils import device

# Dimensions that trigger int32 overflow: B*T*H*K = 4096*576*8*128 = 2.4B > 2^31
B, T, H, K, V = 4096, 576, 8, 128, 128


def _has_enough_gpu_memory(min_gb=20):
    """Check CUDA availability and allocate min_gb to confirm it's usable."""
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.empty(int(min_gb * 1024**3 // 4), dtype=torch.float32, device='cuda')
        del x
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        return False


requires_large_gpu = pytest.mark.skipif(
    not _has_enough_gpu_memory(20),
    reason='Requires CUDA with >= 20GB allocatable memory'
)


@requires_large_gpu
def test_gated_delta_rule_chunk():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32).sigmoid()
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    o, ht = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=h0, output_final_state=True)
    torch.cuda.synchronize()
    assert not o.isnan().any(), "Output contains NaN"


@requires_large_gpu
def test_gated_delta_rule_fused_recurrent():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32).sigmoid()
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    o, ht = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=h0, output_final_state=True)
    torch.cuda.synchronize()
    assert not o.isnan().any(), "Output contains NaN"


@requires_large_gpu
def test_delta_rule_chunk():
    from fla.ops.delta_rule import chunk_delta_rule
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32).sigmoid()
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    o, ht = chunk_delta_rule(q, k, v, beta=beta, initial_state=h0, output_final_state=True)
    torch.cuda.synchronize()
    assert not o.isnan().any(), "Output contains NaN"


@requires_large_gpu
def test_gla_chunk():
    from fla.ops.gla import chunk_gla
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=torch.float32))
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    o, ht = chunk_gla(q, k, v, g, initial_state=h0, output_final_state=True)
    torch.cuda.synchronize()
    assert not o.isnan().any(), "Output contains NaN"


@requires_large_gpu
def test_hgrn_chunk():
    from fla.ops.hgrn import chunk_hgrn
    D = 1024
    x = torch.randn(B, T, D, device=device, dtype=torch.float32)
    g = torch.randn(B, T, D, device=device, dtype=torch.float32)
    h0 = torch.randn(B, D, device=device, dtype=torch.float32)
    o, ht = chunk_hgrn(x, g, initial_state=h0, output_final_state=True)
    torch.cuda.synchronize()
    assert not o.isnan().any(), "Output contains NaN"


@requires_large_gpu
def test_rwkv6_chunk():
    from fla.ops.rwkv6 import chunk_rwkv6
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    w = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    u = torch.randn(H, K, device=device, dtype=torch.float32)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    o, ht = chunk_rwkv6(q, k, v, w, u, initial_state=h0, output_final_state=True)
    torch.cuda.synchronize()
    assert not o.isnan().any(), "Output contains NaN"


@requires_large_gpu
def test_simple_gla_chunk():
    from fla.ops.simple_gla import chunk_simple_gla
    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    o, ht = chunk_simple_gla(q, k, v, g=g, initial_state=h0, output_final_state=True)
    torch.cuda.synchronize()
    assert not o.isnan().any(), "Output contains NaN"
