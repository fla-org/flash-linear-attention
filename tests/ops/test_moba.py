# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.moba import parallel_moba
from fla.utils import assert_close, device

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


def _full_causal_attn(q, k, v, cu_seqlens, max_seqlen):
    # flash_attn_varlen_func wants packed 3-D; strip the B=1 dim for the reference.
    return flash_attn_varlen_func(
        q.squeeze(0), k.squeeze(0), v.squeeze(0),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    ).unsqueeze(0)


# When `topk - 1` is large enough to cover every earlier chunk, MoBA's
# per-query attention set equals (chunk-local causal prefix) ∪ (all earlier
# chunks), which is exactly full causal attention. This gives a deterministic
# reference without writing a separate naive MoBA.
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="flash-attn is required")
@pytest.mark.parametrize(
    ('T', 'H', 'D', 'chunk_size', 'topk'),
    [
        pytest.param(*test, id="T{}-H{}-D{}-C{}-K{}".format(*test))
        for test in [
            (512, 4, 64, 128, 100),
            (1024, 8, 64, 256, 100),
            (2048, 4, 128, 256, 100),
        ]
    ],
)
def test_parallel_moba_matches_full_attn(T, H, D, chunk_size, topk):
    torch.manual_seed(42)
    dtype = torch.float16
    q = torch.randn((1, T, H, D), dtype=dtype, device=device)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

    o_moba = parallel_moba(q, k, v, cu_seqlens, T, chunk_size, topk)
    o_ref = _full_causal_attn(q, k, v, cu_seqlens, T)

    assert o_moba.shape == q.shape
    assert_close(" o", o_ref, o_moba, 0.005)


# `topk=1` exercises the short-circuit branch in `parallel_moba` that
# directly calls `flash_attn_varlen_func` (no chunks selectable beyond the
# local one), which should be a bit-for-bit dense-attention call.
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="flash-attn is required")
def test_parallel_moba_topk1_short_circuits():
    torch.manual_seed(42)
    T, H, D, chunk_size = 1024, 4, 64, 128
    dtype = torch.float16
    q = torch.randn((1, T, H, D), dtype=dtype, device=device)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

    o_moba = parallel_moba(q, k, v, cu_seqlens, T, chunk_size, topk=1)
    o_ref = _full_causal_attn(q, k, v, cu_seqlens, T)

    assert_close(" o", o_ref, o_moba, 0.005)


@pytest.mark.skipif(flash_attn_varlen_func is None, reason="flash-attn is required")
@pytest.mark.parametrize(
    ('cu_seqlens', 'H', 'D', 'chunk_size', 'topk'),
    [
        pytest.param(*test, id="cs{}-H{}-D{}-C{}-K{}".format(*test))
        for test in [
            ([0, 256, 768, 1024], 4, 64, 128, 100),
            ([0, 512, 1536], 8, 64, 256, 100),
            ([0, 200, 600, 1400, 2048], 4, 128, 128, 100),
        ]
    ],
)
def test_parallel_moba_varlen_matches_full_attn(cu_seqlens, H, D, chunk_size, topk):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    dtype = torch.float16
    q = torch.randn((1, T, H, D), dtype=dtype, device=device)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device)
    cu_seqlens_th = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    max_seqlen = int(max(b - a for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)))

    o_moba = parallel_moba(q, k, v, cu_seqlens_th, max_seqlen, chunk_size, topk)
    o_ref = _full_causal_attn(q, k, v, cu_seqlens_th, max_seqlen)

    assert_close(" o", o_ref, o_moba, 0.005)


@pytest.mark.skipif(flash_attn_varlen_func is None, reason="flash-attn is required")
def test_parallel_moba_backward():
    torch.manual_seed(42)
    T, H, D, chunk_size, topk = 512, 4, 64, 128, 100
    dtype = torch.float16
    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, H, D), dtype=dtype, device=device)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

    o_ref = _full_causal_attn(q, k, v, cu_seqlens, T)
    o_ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    o_moba = parallel_moba(q, k, v, cu_seqlens, T, chunk_size, topk)
    o_moba.backward(do)

    assert_close(" o", o_ref, o_moba, 0.005)
    assert_close("dq", ref_dq, q.grad, 0.005)
    assert_close("dk", ref_dk, k.grad, 0.005)
    assert_close("dv", ref_dv, v.grad, 0.005)


# Enforce the fla varlen convention: `cu_seqlens` requires B=1.
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="flash-attn is required")
def test_parallel_moba_rejects_batch_gt_one_with_cu_seqlens():
    T, H, D = 256, 4, 64
    dtype = torch.float16
    q = torch.randn((2, T, H, D), dtype=dtype, device=device)
    k = torch.randn((2, T, H, D), dtype=dtype, device=device)
    v = torch.randn((2, T, H, D), dtype=dtype, device=device)
    cu_seqlens = torch.tensor([0, T, 2 * T], dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="batch size"):
        parallel_moba(q, k, v, cu_seqlens, T, chunk_size=128, topk=4)


# Sanity guard: with a small `topk`, the block-sparse output MUST diverge
# from dense causal attention. Catches regressions where selection silently
# falls back to full attention.
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="flash-attn is required")
def test_parallel_moba_is_sparse_when_topk_small():
    torch.manual_seed(42)
    T, H, D, chunk_size, topk = 1024, 4, 64, 128, 2
    dtype = torch.float16
    q = torch.randn((1, T, H, D), dtype=dtype, device=device)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

    o_moba = parallel_moba(q, k, v, cu_seqlens, T, chunk_size, topk)
    o_ref = _full_causal_attn(q, k, v, cu_seqlens, T)

    assert not torch.allclose(o_moba, o_ref, atol=1e-2, rtol=1e-2), (
        "sparse MoBA should not match full causal attention"
    )
