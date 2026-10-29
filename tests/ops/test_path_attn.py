# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch
import torch.nn.functional as F
from triton.runtime.jit import JITFunction

from fla.ops.path_attn import naive_path_attn
from fla.ops.path_attn.cumprod_householder_bwd import chunk_cumprod_householder_bwd_kernel
from fla.ops.path_attn.cumprod_householder_fwd import chunk_cumprod_householder_fwd_kernel
from fla.ops.path_attn.parallel import parallel_path_attention
from fla.ops.path_attn.parallel_path_bwd_inter_dkv import parallel_path_bwd_dkv_kernel
from fla.ops.path_attn.parallel_path_bwd_inter_dqh import parallel_path_bwd_dq_kernel
from fla.ops.path_attn.transform_q import transform_q_fwd_kernel
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'use_forget_gate', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-use_forget_gate{}-{}".format(*test))
        for test in [
            # SY (2025/07/08): It somehow failed on Hopper with error msg: Aborted (core dumped)
            # (10, 62, 2, 8, 128, True, torch.bfloat16),
            (5, 512, 2, 8, 128, True, torch.bfloat16),
            (3, 1024, 2, 8, 64, True, torch.bfloat16),
            (2, 2000, 1, 4, 64, False, torch.bfloat16),
            (1, 4000, 1, 2, 128, False, torch.bfloat16),
        ]
    ],
)
@pytest.mark.skipif(
    IS_INTEL_ALCHEMIST,
    reason="Intel Triton Failure",
)
def test_parallel(
    B: int,
    H: int,
    HQ: int,
    T: int,
    D: int,
    use_forget_gate: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    w = F.normalize(torch.randn((B, T, H, D), dtype=torch.float, device=device), dim=-1, p=2).requires_grad_(True)
    beta = torch.empty((B, T, H), dtype=torch.float, device=device).uniform_(1.5, 2.0).requires_grad_(True)
    if use_forget_gate:
        g = torch.empty((B, T, HQ), dtype=torch.float, device=device).uniform_(
            0.95, 1).log().requires_grad_(True)
    else:
        g = None
    do = torch.rand((B, T, HQ, D), dtype=dtype, device=device)
    scale = D ** -0.5
    ref = naive_path_attn(q, k, v, w, beta, torch.zeros(B, T, HQ, device=device, dtype=torch.float) if g is None else g, scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    if use_forget_gate:
        ref_dg, g.grad = g.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_db, beta.grad = beta.grad.clone(), None

    tri, _ = parallel_path_attention(q=q, k=k, v=v, w=w, beta=beta, g=g, scale=scale)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    if use_forget_gate:
        tri_dg, g.grad = g.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_db, beta.grad = beta.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.008)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.008)
    if use_forget_gate:
        assert_close("dg", ref_dg, tri_dg, 0.02)
    assert_close("dw", ref_dw, tri_dw, 0.015)
    assert_close("db", ref_db, tri_db, 0.02)


@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'use_forget_gate', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-use_forget_gate{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 4, 128, False, [0, 15, 333, 2048], torch.float16),
            (2, 4, 128, True, [0, 15, 333, 2048], torch.float16),
            (2, 4, 64, True, [0, 841, 889, 4096], torch.float16),
            (2, 4, 64, False, [0, 841, 889, 2000, 3000, 4096], torch.float16),
            (2, 16, 128, True, [0, 500, 1023, 2000, 3000, 4096], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled",
)
@pytest.mark.skipif(
    IS_INTEL_ALCHEMIST,
    reason="Intel Triton Failure",
)
@pytest.mark.smoke
def test_parallel_varlen(
    H: int,
    HQ: int,
    D: int,
    use_forget_gate: bool,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    w = F.normalize(torch.randn((1, T, H, D), dtype=torch.float, device=device), dim=-1, p=2).requires_grad_(True)
    beta = torch.rand((1, T, H), dtype=torch.float, device=device).sigmoid().requires_grad_(True)
    if use_forget_gate:
        g = torch.empty((1, T, HQ), dtype=torch.float, device=device).uniform_(0.95, 1).log().requires_grad_(True)
    else:
        g = None
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)
    scale = D ** -0.5
    ref = torch.zeros(1, T, HQ, D, device=device, dtype=dtype)
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False):
        g_segment = torch.zeros(1, eos - bos, HQ, device=device, dtype=torch.float) if g is None else g[:, bos:eos]
        ref[:, bos:eos] = naive_path_attn(
            q[:, bos:eos], k[:, bos:eos], v[:, bos:eos],
            w[:, bos:eos], beta[:, bos:eos], g_segment, scale,
        )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    if use_forget_gate:
        ref_dg, g.grad = g.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_db, beta.grad = beta.grad.clone(), None
    tri, _ = parallel_path_attention(q=q, k=k, v=v, w=w, beta=beta, g=g, scale=scale, cu_seqlens=cu_seqlens)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    if use_forget_gate:
        tri_dg, g.grad = g.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_db, beta.grad = beta.grad.clone(), None
    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)
    if use_forget_gate:
        assert_close("dg", ref_dg, tri_dg, 0.005)
    assert_close("dw", ref_dw, tri_dw, 0.005)
    assert_close("db", ref_db, tri_db, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'K', 'V', 'cu_seqlens'),
    [
        pytest.param(2, 1024, 64, 128, None, id="dense-asymmetric"),
        pytest.param(1, 2048, 64, 128, [0, 1024, 2048], id="varlen-asymmetric"),
        pytest.param(2, 1024, 64, 64, None, id="dense-equal-width"),
        pytest.param(2, 512, 64, 128, None, id="dense-short"),
    ],
)
@pytest.mark.skipif(
    IS_INTEL_ALCHEMIST,
    reason="Intel Triton Failure",
)
def test_parallel_dv_isolation(B: int, T: int, K: int, V: int, cu_seqlens: list[int] | None):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, 1, K), dtype=torch.bfloat16, device=device).requires_grad_(True)
    k = torch.randn((B, T, 1, K), dtype=torch.bfloat16, device=device).requires_grad_(True)
    v = torch.randn((B, T, 1, V), dtype=torch.bfloat16, device=device).requires_grad_(True)
    w = F.normalize(torch.randn((B, T, 1, K), dtype=torch.float, device=device), dim=-1, p=2).requires_grad_(True)
    beta = torch.rand((B, T, 1), dtype=torch.float, device=device).sigmoid().requires_grad_(True)
    cu_seqlens_tensor = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    out, _ = parallel_path_attention(q=q, k=k, v=v, w=w, beta=beta, cu_seqlens=cu_seqlens_tensor)
    do = torch.zeros_like(out)
    if cu_seqlens is None:
        target = do[1]
        untouched = (0, slice(None))
    else:
        target = do[0, cu_seqlens[1]:]
        untouched = (0, slice(0, cu_seqlens[1]))
    target.copy_(torch.linspace(0.25, 1.0, target.numel(), dtype=target.dtype, device=device).reshape_as(target))

    _, _, dv, _, _ = torch.autograd.grad(out, (q, k, v, w, beta), grad_outputs=do)
    assert torch.isfinite(dv).all()
    assert torch.count_nonzero(dv[untouched]).item() == 0


@pytest.mark.parametrize(('kernel', 'name'), [
    (chunk_cumprod_householder_fwd_kernel, 'T'),
    (chunk_cumprod_householder_bwd_kernel, 'T'),
    (transform_q_fwd_kernel, 'NUM_BLOCKS'),
    (parallel_path_bwd_dkv_kernel, 'NUM_BLOCKS'),
    (parallel_path_bwd_dq_kernel, 'NUM_BLOCKS'),
])
def test_parallel_backward_kernels_take_runtime_lengths(kernel, name):
    """T and NUM_BLOCKS follow the (longest) sequence length; as constexprs, every new length compiles the kernel again."""
    jit = kernel if isinstance(kernel, JITFunction) else kernel.fn
    param = next(param for param in jit.params if param.name == name)
    assert not param.is_constexpr
    assert param.do_not_specialize
