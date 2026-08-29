# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import pytest
import torch

from fla.ops.deltaformer import deltaformer_attn
from fla.ops.deltaformer.naive import naive_deltaformer_attn, tril_softmax
from fla.ops.deltaformer.parallel import ParallelDeltaformerFunction
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device, find_spec_cached

# Only the public deltaformer_attn needs flash-attn (to consume u); the u computation itself
# does not, so the mark sits on individual tests rather than the module. A module-level
# importorskip would also exit pytest with code 5 (no tests collected), which breaks CI's
# per-file `pytest "$f" || exit 1` loop.
requires_flash_attn = pytest.mark.skipif(
    find_spec_cached("flash_attn") is None,
    reason="deltaformer_attn requires flash-attn (`pip install flash-attn --no-build-isolation`).",
)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'C', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-C{}-{}".format(*test))
        for test in [
            (2, 128, 2, 64, 512, torch.float16),
            (1, 256, 4, 64, 512, torch.float16),
            (2, 512, 4, 64, 512, torch.float16),
            (4, 1024, 4, 128, 512, torch.float16),
            # chunk sizes the autotuned BLOCK_T values (64, 32) do not divide, so a key
            # tile straddles the chunk boundary
            (2, 192, 2, 64, 48, torch.float16),
            (1, 384, 2, 64, 96, torch.float16),
        ]
    ],
)
@requires_flash_attn
@pytest.mark.skipif(
    IS_INTEL_ALCHEMIST,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM.",
)
def test_deltaformer_attn(
    B: int,
    T: int,
    H: int,
    D: int,
    C: int,
    dtype: torch.dtype,
):
    """
    Test DeltaFormer pre-attention by comparing fused implementation with naive reference.
    """
    torch.manual_seed(42)

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    beta = torch.randn((B, T, H), dtype=dtype, device=device).sigmoid().requires_grad_(True)

    do = torch.randn((B, T, H, D), dtype=dtype, device=device)

    ref = naive_deltaformer_attn(q, k, v, beta)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dbeta, beta.grad = beta.grad.clone(), None

    tri = deltaformer_attn(q, k, v, beta, C=C)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dbeta, beta.grad = beta.grad.clone(), None

    assert_close('o', ref, tri, 0.006)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('dbeta', ref_dbeta, tri_dbeta, 0.008)


def naive_deltaformer_u(q, k, v, beta):
    """
    Stage 1 of naive_deltaformer_attn (which only returns o), sequence-first layout:
    u[t] = v[t] - beta[t] * sum_{j<t} softmax(q[t] @ k[:t]^T)[j] * u[j], computed in fp32.
    """
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    betaf = beta.float().transpose(1, 2)
    scores = qf @ kf.transpose(-1, -2) * (1.0 / math.sqrt(q.shape[-1]))
    probs = tril_softmax(scores, strict=True)
    us = []
    for t in range(q.shape[1]):
        u_t = vf[:, :, t]
        if t > 0:
            u_prev = torch.stack(us, dim=-2)
            u_t = u_t - betaf[:, :, t, None] * (probs[:, :, t, :t, None] * u_prev).sum(-2)
        us.append(u_t)
    return torch.stack(us, dim=2).transpose(1, 2).to(q.dtype)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'C', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-C{}-{}".format(*test))
        for test in [
            # chunk sizes the autotuned BLOCK_T values (64, 32) do not divide, so a key
            # tile straddles the chunk boundary; C=64 is the aligned control
            (2, 192, 2, 64, 16, torch.float16),
            (2, 192, 2, 64, 48, torch.float16),
            (2, 192, 2, 64, 64, torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    IS_INTEL_ALCHEMIST,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM.",
)
def test_parallel_deltaformer_u(
    B: int,
    T: int,
    H: int,
    D: int,
    C: int,
    dtype: torch.dtype,
):
    """
    Test the u-computation stage (ParallelDeltaformerFunction) against the naive recurrence.

    Unlike test_deltaformer_attn this needs no flash-attn, so it also runs on CI runners
    without a flash-attn build.
    """
    torch.manual_seed(42)

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    beta = torch.randn((B, T, H), dtype=dtype, device=device).sigmoid().requires_grad_(True)

    du = torch.randn((B, T, H, D), dtype=dtype, device=device)

    ref = naive_deltaformer_u(q, k, v, beta)
    ref.backward(du)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dbeta, beta.grad = beta.grad.clone(), None

    tri = ParallelDeltaformerFunction.apply(q, k, v, beta, C, None)
    tri.backward(du)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dbeta, beta.grad = beta.grad.clone(), None

    assert_close('u', ref, tri, 0.006)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('dbeta', ref_dbeta, tri_dbeta, 0.008)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 64, [0, 63], torch.float16),
            (4, 64, [0, 256, 500, 1000], torch.float16),
            (4, 128, [0, 15, 100, 300, 1200, 2000], torch.float16),
            (2, 128, [0, 100, 123, 300, 500, 800, 1000, 1500, 2048], torch.float16),
        ]
    ],
)
@requires_flash_attn
@pytest.mark.skipif(
    IS_INTEL_ALCHEMIST,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM.",
)
def test_deltaformer_attn_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    beta = torch.randn((1, T, H), dtype=dtype, device=device).sigmoid().requires_grad_()

    do = torch.randn_like(q)

    refs = []
    for i in range(N):
        ref = naive_deltaformer_attn(
            q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            beta[:, cu_seqlens[i]:cu_seqlens[i+1]],
        )
        refs.append(ref)
    ref = torch.cat(refs, dim=1)

    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dbeta, beta.grad = beta.grad.clone(), None

    tri = deltaformer_attn(q, k, v, beta, cu_seqlens=cu_seqlens)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dbeta, beta.grad = beta.grad.clone(), None

    assert_close('o', ref, tri, 0.006)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('dbeta', ref_dbeta, tri_dbeta, 0.008)
