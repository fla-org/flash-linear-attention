# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.stickbreaking_attn import naive_stickbreaking_attn, parallel_stickbreaking_attn
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, check_shared_mem, device

TOL = {torch.float16: 0.005, torch.bfloat16: 0.02}


@pytest.mark.parametrize('attend_current', [False, True])
def test_naive_matches_definition(attend_current: bool):
    torch.manual_seed(42)
    B, T, H, D = 2, 16, 2, 8
    q = torch.randn((B, T, H, D), dtype=torch.float64, device=device)
    k = torch.randn((B, T, H, D), dtype=torch.float64, device=device)
    v = torch.randn((B, T, H, D), dtype=torch.float64, device=device)
    scale = D ** -0.5

    # A_ij = beta_ij * prod_l (1 - beta_il) over the visible keys l after j, written out key by key
    beta = torch.einsum('bqhd,bkhd->bhqk', q, k).mul(scale).sigmoid()
    att = torch.zeros_like(beta)
    for i in range(T):
        stick = torch.ones_like(beta[..., i, 0])
        for j in range(i if attend_current else i - 1, -1, -1):
            att[..., i, j] = beta[..., i, j] * stick
            stick = stick * (1 - beta[..., i, j])
    ref_o = torch.einsum('bhqk,bkhd->bqhd', att, v)
    ref_rem = (1 - att.sum(-1)).transpose(1, 2)

    o, rem = naive_stickbreaking_attn(q, k, v, scale=scale, attend_current=attend_current)
    torch.testing.assert_close(o, ref_o, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rem, ref_rem, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'K', 'V', 'scale', 'attend_current'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-K{}-V{}-scale{}-attend_current{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 64, 1.0, False),
            (3, 111, 2, 2, 100, 100, 1.0, True),
            (3, 127, 2, 8, 60, 60, 0.1, False),
            (2, 1024, 2, 8, 64, 128, 0.1, True),
            (2, 1024, 2, 2, 128, 128, 0.1, False),
            (2, 1024, 1, 4, 256, 64, 0.1, True),
        ]
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_parallel(
    B: int,
    T: int,
    H: int,
    HQ: int,
    K: int,
    V: int,
    scale: float,
    attend_current: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if not check_shared_mem('hopper') and max(K, V) > 128:
        pytest.skip("Skipping test because global shared memory is not available")

    q = torch.randn((B, T, HQ, K), dtype=dtype, device=device)
    k = torch.randn((B, T, H, K), dtype=dtype, device=device)
    v = torch.randn((B, T, H, V), dtype=dtype, device=device)

    ref_o, ref_rem = naive_stickbreaking_attn(q, k, v, scale=scale, attend_current=attend_current)
    tri_o, tri_rem = parallel_stickbreaking_attn(q, k, v, scale=scale, attend_current=attend_current)

    assert_close("  o", ref_o, tri_o, TOL[dtype])
    assert_close("rem", ref_rem, tri_rem, TOL[dtype])


@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'cu_seqlens', 'attend_current'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-cu_seqlens{}-attend_current{}".format(*test))
        for test in [
            (2, 2, 64, [0, 15], False),
            (2, 8, 64, [0, 256, 500, 1000], True),
            (2, 2, 100, [0, 15, 100, 300, 1200, 2000], False),
        ]
    ],
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
    cu_seqlens: list[int],
    attend_current: bool,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    dtype = torch.float16
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device)

    ref_o = q.new_empty(1, T, HQ, D)
    ref_rem = q.new_empty(1, T, HQ)
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False):
        ref_o[:, bos:eos], ref_rem[:, bos:eos] = naive_stickbreaking_attn(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            attend_current=attend_current,
        )

    tri_o, tri_rem = parallel_stickbreaking_attn(q=q, k=k, v=v, attend_current=attend_current, cu_seqlens=cu_seqlens)

    assert_close("  o", ref_o, tri_o, 0.005)
    assert_close("rem", ref_rem, tri_rem, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'cu_seqlens', 'attend_current'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-cu_seqlens{}-attend_current{}".format(*test))
        for test in [
            (2, 1024, 2, 2, 64, None, False),
            (2, 1024, 1, 4, 128, None, True),
            (1, 2000, 2, 2, 64, [0, 15, 100, 300, 1200, 2000], True),
        ]
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_parallel_long_range(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    cu_seqlens: list[int] | None,
    attend_current: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    # zero-mean logits use up the stick within a few dozen keys, hiding the key blocks past the first one below the diagonal;
    # a negative logit offset keeps sigmoid(z) small, so the stick lasts across the sequence
    q[..., 0], k[..., 0] = 8, -8
    scale = 0.1

    if cu_seqlens is None:
        ref_o, ref_rem = naive_stickbreaking_attn(q, k, v, scale=scale, attend_current=attend_current)
    else:
        ref_o, ref_rem = q.new_empty(B, T, HQ, D), q.new_empty(B, T, HQ)
        for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False):
            ref_o[:, bos:eos], ref_rem[:, bos:eos] = naive_stickbreaking_attn(
                q=q[:, bos:eos],
                k=k[:, bos:eos],
                v=v[:, bos:eos],
                scale=scale,
                attend_current=attend_current,
            )
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    tri_o, tri_rem = parallel_stickbreaking_attn(q, k, v, scale=scale, attend_current=attend_current, cu_seqlens=cu_seqlens)

    assert_close("  o", ref_o, tri_o, TOL[dtype])
    assert_close("rem", ref_rem, tri_rem, TOL[dtype])


def test_parallel_backward_not_implemented():
    q = torch.randn((1, 64, 2, 64), dtype=torch.float16, device=device).requires_grad_()
    k = torch.randn((1, 64, 2, 64), dtype=torch.float16, device=device).requires_grad_()
    v = torch.randn((1, 64, 2, 64), dtype=torch.float16, device=device).requires_grad_()

    o, _ = parallel_stickbreaking_attn(q, k, v)
    with pytest.raises(NotImplementedError, match="Backward pass is not implemented"):
        o.sum().backward()


@pytest.mark.parametrize("op", [naive_stickbreaking_attn, parallel_stickbreaking_attn], ids=["naive", "parallel"])
@pytest.mark.parametrize(("HQ", "H"), [(3, 2), (1, 2), (2, 0)], ids=["remainder", "fewer-query-heads", "zero-kv-heads"])
def test_parallel_rejects_invalid_gqa_head_counts(op, HQ, H):
    q = torch.empty(1, 1, HQ, 16)
    k = torch.empty(1, 1, H, 16)
    v = torch.empty_like(k)

    with pytest.raises(ValueError, match="must be divisible"):
        op(q=q, k=k, v=v)


def test_parallel_rejects_batched_varlen():
    q = torch.empty(2, 8, 2, 16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)

    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        parallel_stickbreaking_attn(q=q, k=k, v=v, cu_seqlens=cu_seqlens)
