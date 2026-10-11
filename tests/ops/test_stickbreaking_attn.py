# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.stickbreaking_attn import naive_stickbreaking_attn, parallel_stickbreaking_attn
from fla.utils import FLA_DISABLE_TENSOR_CACHE, IS_AMD, IS_INTEL_ALCHEMIST, IS_NVIDIA, assert_close, check_shared_mem, device

TOL = {torch.float16: 0.005, torch.bfloat16: 0.02}


def _naive_varlen(q, k, v, cu_seqlens, **kwargs):
    o, rem = q.new_empty(*q.shape[:-1], v.shape[-1]), q.new_empty(q.shape[:-1])
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False):
        o[:, bos:eos], rem[:, bos:eos] = naive_stickbreaking_attn(q=q[:, bos:eos], k=k[:, bos:eos], v=v[:, bos:eos], **kwargs)
    return o, rem


def _forward_backward(op, q, k, v, do, drem, **kwargs):
    o, rem = op(q=q, k=k, v=v, **kwargs)
    torch.autograd.backward((o, rem), (do, drem))
    grads = [x.grad.clone() for x in (q, k, v)]
    q.grad = k.grad = v.grad = None
    return o, rem, *grads


def _assert_all_close(ref, tri, ratio):
    for name, x, y in zip(("  o", "rem", " dq", " dk", " dv"), ref, tri, strict=True):
        assert_close(prefix=name, ref=x, tri=y, ratio=ratio)


@pytest.mark.parametrize('attend_current', [False, True])
def test_naive_matches_definition(attend_current: bool):
    torch.manual_seed(42)
    B, T, H, D = 2, 16, 2, 8
    q = torch.randn((B, T, H, D), dtype=torch.float64, device=device)
    k = torch.randn((B, T, H, D), dtype=torch.float64, device=device)
    v = torch.randn((B, T, H, D), dtype=torch.float64, device=device)
    scale = D ** -0.5

    beta = torch.einsum('bqhd,bkhd->bhqk', q, k).mul(scale).sigmoid()
    att = torch.zeros_like(beta)
    for i in range(T):
        stick = torch.ones_like(beta[..., i, 0])
        for j in range(i if attend_current else i - 1, -1, -1):
            att[..., i, j] = beta[..., i, j] * stick
            stick = stick * (1 - beta[..., i, j])
    ref_o = torch.einsum('bhqk,bkhd->bqhd', att, v)
    ref_rem = (1 - att.sum(-1)).transpose(1, 2)

    # both sides compute in fp64, so the TF32 settings of fp32 matmuls cannot affect the result
    o, rem = naive_stickbreaking_attn(q=q, k=k, v=v, scale=scale, attend_current=attend_current)
    torch.testing.assert_close(o, ref_o, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(rem, ref_rem, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not IS_NVIDIA, reason="TF32 is an NVIDIA tensor-core mode")
def test_naive_ignores_tf32():
    torch.manual_seed(42)
    # large enough that cuBLAS picks TF32 kernels for fp32 matmuls; the definition test's shape is too small for that
    B, T, H, D = 2, 256, 2, 64
    q = torch.randn((B, T, H, D), dtype=torch.float64, device=device)
    k = torch.randn((B, T, H, D), dtype=torch.float64, device=device)
    v = torch.randn((B, T, H, D), dtype=torch.float64, device=device)
    x = torch.randn((B * H, T, D), device=device)
    y = torch.randn((B * H, D, T), device=device)

    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    outs = []
    try:
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            outs.append((torch.bmm(x, y), *naive_stickbreaking_attn(q=q, k=k, v=v)))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    (xy, o, rem), (xy_tf32, o_tf32, rem_tf32) = outs
    if torch.equal(xy, xy_tf32):
        pytest.skip("TF32 does not change fp32 matmuls on this device or configuration")
    torch.testing.assert_close(o_tf32, o, rtol=0, atol=0)
    torch.testing.assert_close(rem_tf32, rem, rtol=0, atol=0)


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
            (2, 1024, 1, 4, 64, 256, 0.1, False),
        ]
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
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
    if not check_shared_mem(arch='hopper') and max(K, V) > 128:
        pytest.skip("This test requires Hopper-class shared memory for head dimensions above 128.")

    q = torch.randn((B, T, HQ, K), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, V), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((B, T, HQ, V), dtype=dtype, device=device)
    drem = torch.randn((B, T, HQ), dtype=dtype, device=device)

    ref = _forward_backward(
        op=naive_stickbreaking_attn,
        q=q,
        k=k,
        v=v,
        do=do,
        drem=drem,
        scale=scale,
        attend_current=attend_current,
    )
    tri = _forward_backward(
        op=parallel_stickbreaking_attn,
        q=q,
        k=k,
        v=v,
        do=do,
        drem=drem,
        scale=scale,
        attend_current=attend_current,
    )
    _assert_all_close(ref=ref, tri=tri, ratio=TOL[dtype])


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
@pytest.mark.skipif(IS_INTEL_ALCHEMIST, reason="Intel Triton Failure")
@pytest.mark.smoke
def test_parallel_varlen(H: int, HQ: int, D: int, cu_seqlens: list[int], attend_current: bool):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    dtype = torch.float16
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)
    drem = torch.randn((1, T, HQ), dtype=dtype, device=device)

    kwargs = dict(cu_seqlens=cu_seqlens, attend_current=attend_current)

    ref = _forward_backward(op=_naive_varlen, q=q, k=k, v=v, do=do, drem=drem, **kwargs)
    tri = _forward_backward(op=parallel_stickbreaking_attn, q=q, k=k, v=v, do=do, drem=drem, **kwargs)
    _assert_all_close(ref=ref, tri=tri, ratio=0.005)


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
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
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
    # negative logits keep the stick alive across key blocks, exposing long-range errors
    q[..., 0], k[..., 0] = 8, -8
    q, k, v = (x.requires_grad_() for x in (q, k, v))
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    drem = torch.randn((B, T, HQ), dtype=dtype, device=device)
    kwargs = dict(scale=0.1, attend_current=attend_current)

    if cu_seqlens is None:
        ref = _forward_backward(op=naive_stickbreaking_attn, q=q, k=k, v=v, do=do, drem=drem, **kwargs)
    else:
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        ref = _forward_backward(op=_naive_varlen, q=q, k=k, v=v, do=do, drem=drem, cu_seqlens=cu_seqlens, **kwargs)
    tri = _forward_backward(op=parallel_stickbreaking_attn, q=q, k=k, v=v, do=do, drem=drem, cu_seqlens=cu_seqlens, **kwargs)
    _assert_all_close(ref=ref, tri=tri, ratio=TOL[dtype])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
def test_parallel_saturated_two_tokens(dtype: torch.dtype):
    # query 1 sees only key 0, and a logit of 11 puts softplus2 on its linear branch where beta rounds to 1
    z, D = 11., 16
    q, k, v, do = (torch.zeros((1, 2, 1, D), dtype=dtype, device=device) for _ in range(4))
    q[0, 1, 0, 0], k[0, 0, 0, 0], v[0, 0, 0, 0], do[0, 1, 0, 0] = z, 1, 4, 4
    q, k, v = (x.requires_grad_() for x in (q, k, v))
    drem = torch.zeros((1, 2, 1), dtype=dtype, device=device)
    _, _, dq, dk, _ = _forward_backward(op=parallel_stickbreaking_attn, q=q, k=k, v=v, do=do, drem=drem, scale=1.)

    # dz = sigmoid(z) * sigmoid(-z) * (do_1 . v_0 - drem_1), and dq_1 = dz * k_0, dk_0 = dz * q_1
    dz = torch.tensor(z, dtype=torch.float64).sigmoid() * torch.tensor(-z, dtype=torch.float64).sigmoid() * 16
    ref_dq, ref_dk = dq.new_zeros(dq.shape, dtype=torch.float64), dk.new_zeros(dk.shape, dtype=torch.float64)
    ref_dq[0, 1, 0, 0], ref_dk[0, 0, 0, 0] = dz, dz * z
    torch.testing.assert_close(dq.double(), ref_dq, rtol=TOL[dtype], atol=0)
    torch.testing.assert_close(dk.double(), ref_dk, rtol=TOL[dtype], atol=0)


@pytest.mark.parametrize(
    ('z', 'dtype'),
    [
        pytest.param(z, dtype, id=f"z{z}-{name}")
        for dtype, name, zs in [
            # past z = 15 the fp16 logit gradients cast for the key matmuls are too deep in the subnormal range
            (torch.float16, 'fp16', [5., 10., 11., 12., 15.]),
            (torch.bfloat16, 'bf16', [5., 10., 11., 12., 15., 20., 30.]),
        ]
        for z in zs
    ],
)
@pytest.mark.parametrize('cu_seqlens', [None, [0, 3, 70, 100]], ids=['dense', 'varlen'])
@pytest.mark.parametrize('attend_current', [False, True])
def test_parallel_saturated_logits(z: float, dtype: torch.dtype, cu_seqlens: list[int] | None, attend_current: bool):
    torch.manual_seed(42)
    T, H, HQ, D = 100, 1, 2, 32
    # every logit is z plus a small jitter, so all rows share one gradient scale and the ratio cannot hide saturated rows
    q, k = 0.25 * torch.randn((1, T, HQ, D), device=device), 0.25 * torch.randn((1, T, H, D), device=device)
    q[..., 0], k[..., 0] = z, 1
    v, do = 4 * torch.randn((1, T, H, D), device=device), 4 * torch.randn((1, T, HQ, D), device=device)
    q, k, v, do = (x.to(dtype) for x in (q, k, v, do))
    drem = torch.randn((1, T, HQ), dtype=dtype, device=device)
    kwargs = dict(scale=1., attend_current=attend_current)
    if cu_seqlens is not None:
        kwargs['cu_seqlens'] = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # fp32 autograd through the naive reference cancels at saturation, so the reference runs in fp64
    ref_q, ref_k, ref_v = (x.double().requires_grad_() for x in (q, k, v))
    ref_op = naive_stickbreaking_attn if cu_seqlens is None else _naive_varlen
    ref = _forward_backward(op=ref_op, q=ref_q, k=ref_k, v=ref_v, do=do.double(), drem=drem.double(), **kwargs)
    q, k, v = (x.requires_grad_() for x in (q, k, v))
    tri = _forward_backward(op=parallel_stickbreaking_attn, q=q, k=k, v=v, do=do, drem=drem, **kwargs)
    # compare in fp64 without the CI warning path of assert_close, which would let tiny wrong gradients pass,
    # against the reference rounded to dtype so that rounding the outputs, e.g. a subnormal rem, does not count as error
    for name, x, y in zip(("o", "rem", "dq", "dk", "dv"), ref, tri, strict=True):
        x = x.to(dtype).double()
        error_rate = ((x - y.double()).norm() / x.norm()).item()
        assert error_rate < TOL[dtype], f"{name} ratio: {error_rate:.6f}"


@pytest.mark.parametrize('cu_seqlens', [None, [0, 100, 1100, 2000]], ids=['dense', 'varlen'])
def test_parallel_backward_deterministic(cu_seqlens: list[int] | None):
    torch.manual_seed(42)
    T, H, HQ, D, dtype = 2000, 2, 4, 64, torch.bfloat16
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)
    drem = torch.randn((1, T, HQ), dtype=dtype, device=device)
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    first = _forward_backward(op=parallel_stickbreaking_attn, q=q, k=k, v=v, do=do, drem=drem, cu_seqlens=cu_seqlens)
    second = _forward_backward(op=parallel_stickbreaking_attn, q=q, k=k, v=v, do=do, drem=drem, cu_seqlens=cu_seqlens)
    for x, y in zip(first, second, strict=True):
        assert torch.equal(x, y)


@pytest.mark.parametrize('cu_seqlens', [None, [0, 100, 300, 500]], ids=['dense', 'varlen'])
@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD), reason="CUDA graph capture requires a CUDA/HIP device")
@pytest.mark.skipif(FLA_DISABLE_TENSOR_CACHE, reason="graph capture relies on the cached sequence metadata")
def test_parallel_cuda_graph(cu_seqlens: list[int] | None):
    torch.manual_seed(42)
    T, H, HQ, D, dtype = 500, 2, 4, 64, torch.float16
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)
    drem = torch.randn((1, T, HQ), dtype=dtype, device=device)
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    def step():
        o, rem = parallel_stickbreaking_attn(q=q, k=k, v=v, cu_seqlens=cu_seqlens)
        return o, rem, *torch.autograd.grad((o, rem), (q, k, v), (do, drem))

    # warm up on a side stream to finish autotuning and fill the sequence metadata cache before capture
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tri = step()

    # new inputs in the captured buffers check that the replay reads them
    with torch.no_grad():
        for x in (q, k, v, do, drem):
            x.copy_(torch.randn_like(x))
    graph.replay()
    ref = step()
    for x, y in zip(ref, tri, strict=True):
        assert torch.equal(x, y)


@pytest.mark.parametrize("op", [naive_stickbreaking_attn, parallel_stickbreaking_attn], ids=["naive", "parallel"])
@pytest.mark.parametrize(("HQ", "H"), [(3, 2), (1, 2), (2, 0)], ids=["remainder", "fewer-query-heads", "zero-kv-heads"])
def test_parallel_rejects_invalid_gqa_head_counts(op, HQ, H):
    q = torch.empty(1, 1, HQ, 16)
    k = torch.empty(1, 1, H, 16)
    v = torch.empty_like(k)

    with pytest.raises(ValueError, match="must be divisible"):
        op(q=q, k=k, v=v)


@pytest.mark.parametrize("op", [naive_stickbreaking_attn, parallel_stickbreaking_attn], ids=["naive", "parallel"])
@pytest.mark.parametrize(
    ("q_shape", "k_shape", "v_shape"),
    [
        ((1, 1, 2, 16), (1, 8, 2, 16), (1, 8, 2, 16)),
        ((2, 8, 2, 16), (1, 8, 2, 16), (1, 8, 2, 16)),
        ((1, 8, 2, 16), (1, 8, 2, 16), (1, 4, 2, 16)),
        ((1, 8, 2, 16), (1, 8, 2, 16), (1, 8, 1, 16)),
        ((1, 8, 2, 32), (1, 8, 2, 16), (1, 8, 2, 16)),
    ],
    ids=["one-query-eight-keys", "batch", "value-length", "value-heads", "key-dim"],
)
def test_parallel_rejects_mismatched_shapes(op, q_shape, k_shape, v_shape):
    with pytest.raises(ValueError, match="Expected q, k and v of shapes"):
        op(q=torch.empty(q_shape), k=torch.empty(k_shape), v=torch.empty(v_shape))


@pytest.mark.parametrize(("K", "V", "match"), [(257, 64, "key dimension"), (64, 257, "value dimension")], ids=["K", "V"])
def test_parallel_rejects_head_dims_above_256(K, V, match):
    q = torch.empty(1, 8, 2, K)
    k = torch.empty_like(q)
    v = torch.empty(1, 8, 2, V)

    with pytest.raises(ValueError, match=match):
        parallel_stickbreaking_attn(q=q, k=k, v=v)


def test_parallel_rejects_batched_varlen():
    q = torch.empty(2, 8, 2, 16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)

    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        parallel_stickbreaking_attn(q=q, k=k, v=v, cu_seqlens=cu_seqlens)
