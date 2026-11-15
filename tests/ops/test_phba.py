# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.phba import SequenceLengthPlan, chunk_phba, flash_attention
from fla.ops.phba.naive import HostLayout, highest_precision, reference_a_chunkwise, rstate_from_rtok
from fla.ops.phba.routing import build_routes
from fla.utils import IS_NVIDIA, assert_close, device, find_spec_cached

pytestmark = pytest.mark.skipif(
    not IS_NVIDIA or find_spec_cached("flash_attn") is None,
    reason="PHBA requires NVIDIA CUDA and flash-attn2.8.3",
)


@pytest.mark.parametrize(
    ("lengths", "H", "D", "M", "chunk_size", "topk", "dtype"),
    [
        pytest.param((67,), 2, 32, 16, 32, 2, torch.float16, id="short-tail-fp16", marks=pytest.mark.smoke),
        pytest.param((129,), 2, 64, 32, 64, 4, torch.bfloat16, id="chunk-boundary-bf16", marks=pytest.mark.smoke),
        pytest.param(
            (65, 1, 33), 2, 72, 80, 32, 2, torch.float16,
            id="varlen-one-token-nonpower-width",
            marks=pytest.mark.smoke,
        ),
        pytest.param((257, 129, 1), 2, 128, 128, 128, 8, torch.bfloat16, id="varlen-eight-routes", marks=pytest.mark.smoke),
        pytest.param((33,), 1, 16, 16, 32, 1, torch.float16, id="no-token-history"),
        pytest.param((65, 65), 1, 32, 17, 64, 3, torch.bfloat16, id="nonpower-slots"),
        pytest.param((513,), 2, 128, 128, 256, 4, torch.bfloat16, id="chunk256"),
        pytest.param((1025,), 2, 128, 64, 512, 2, torch.float16, id="chunk512"),
    ],
)
def test_chunk(lengths, H, D, M, chunk_size, topk, dtype):
    torch.manual_seed(42)
    T = sum(lengths)
    inputs = [torch.randn(1, T, H, D, dtype=dtype, device=device) * 0.1 for _ in range(5)]
    inputs.append(torch.randn(1, T, H, M, dtype=dtype, device=device) * 0.3)
    actual = [x.clone().requires_grad_() for x in inputs]
    expected = [x.clone().requires_grad_() for x in inputs]
    plan = SequenceLengthPlan.from_host(lengths, device)
    layout, rtok, rstate = build_routes(
        q_rope=inputs[3].squeeze(0),
        k_rope=inputs[4].squeeze(0),
        length_plan=plan,
        chunk_size=chunk_size,
        topk=topk,
    )
    do = torch.randn_like(inputs[0])
    with highest_precision():
        tri, _ = chunk_phba(
            q=actual[0],
            k=actual[1],
            v=actual[2],
            q_rope=actual[3],
            k_rope=actual[4],
            gate=actual[5],
            chunk_size=chunk_size,
            topk=topk,
            length_plan=plan,
        )
        refs = [x.squeeze(0) for x in expected]
        ref = reference_a_chunkwise(
            q_raw=refs[0],
            q_rope=refs[3],
            k_rope=refs[4],
            k_raw=refs[1],
            v=refs[2],
            a_storage=refs[5],
            tau=8.0,
            layout=HostLayout(chunk_size, layout.block_ptr_host, layout.seq_block_ptr_host),
            rtok=rtok,
            rstate=rstate,
            local_impl="flash_attn",
            fa_deterministic=True,
        )[0].unsqueeze(
            0,
        )
        tri_grads = torch.autograd.grad(tri, actual, do)
        ref_grads = torch.autograd.grad(ref, expected, do)
    output_limit = 0.03 if dtype == torch.bfloat16 else 0.006
    gradient_limit = 0.06 if dtype == torch.bfloat16 else 0.015
    assert_close("output", ref.float(), tri.float(), output_limit)
    for name, reference, result in zip(("dq", "dk", "dv", "dq_rope", "dk_rope", "dgate"), ref_grads, tri_grads):
        assert torch.isfinite(result).all()
        assert_close(name, reference.float(), result.float(), gradient_limit)
        relative_max = (reference.float() - result.float()).abs().max() / reference.float().abs().max().clamp_min(1e-5)
        assert relative_max < gradient_limit


@pytest.mark.parametrize(("chunk_size", "topk"), [(16, 2), (64, 17)], ids=["unsupported-chunk", "too-many-routes"])
def test_rejected_geometry(chunk_size, topk):
    x = torch.zeros(1, 33, 1, 16, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_phba(q=x, k=x, v=x, q_rope=x, k_rope=x, gate=x, chunk_size=chunk_size, topk=topk)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_prefix_final_state(dtype):
    from fla.ops.phba.naive import gate_from_storage, local_cumsum, prefix_forward

    torch.manual_seed(42)
    lengths = (65, 17)
    plan = SequenceLengthPlan.from_host(lengths, device)
    tensors = [torch.randn(1, sum(lengths), 1, 32, device=device, dtype=dtype) * 0.1 for _ in range(6)]
    _, final = chunk_phba(
        q=tensors[0],
        k=tensors[1],
        v=tensors[2],
        q_rope=tensors[3],
        k_rope=tensors[4],
        gate=tensors[5],
        chunk_size=32,
        topk=2,
        length_plan=plan,
        output_final_state=True,
    )
    host = HostLayout.from_lengths(lengths, 32)
    g, s = gate_from_storage(tensors[5].squeeze(0), 8.0, dtype)
    with highest_precision():
        cumulative = local_cumsum(g, host)
        expected = [prefix_forward(s, x.squeeze(0), cumulative, host)[1] for x in tensors[1:3]]
    for name, reference, actual in zip(("final_k", "final_v"), expected, final):
        assert actual.dtype == torch.float32 and not actual.requires_grad
        assert torch.isfinite(actual).all()
        assert_close(name, reference, actual, 0.006)


def test_naive_formula_gradients():
    from fla.ops.phba.naive import naive_phba

    torch.manual_seed(42)
    values = [torch.randn(1, 35, 1, 8, dtype=torch.float64, device=device) * 0.1 for _ in range(6)]
    leaves = [x.requires_grad_() for x in values]
    token_routes = torch.full((1, 35, 1), -1, device=device, dtype=torch.int32)
    token_routes[:, 32:, :] = 0
    state_routes = torch.full((1, 35, 2), -1, device=device, dtype=torch.int32)
    state_routes[:, 32:, 0] = 1

    def formula(*inputs):
        return naive_phba(
            q=inputs[0],
            k=inputs[1],
            v=inputs[2],
            q_rope=inputs[3],
            k_rope=inputs[4],
            gate=inputs[5],
            lengths=(35,),
            token_routes=token_routes,
            state_routes=state_routes,
            chunk_size=32,
        )

    assert torch.autograd.gradcheck(formula, tuple(leaves), fast_mode=True)


@pytest.mark.parametrize("topk", [1, 4], ids=["no-history", "history-and-tail"])
def test_routing_and_metadata(topk):
    from fla.ops.phba.metadata import build_direct_metadata, canonical_rho
    from fla.ops.phba.naive import (
        centroid_reference,
        metadata_reference,
        route_reference,
    )
    from fla.ops.phba.routing import compute_block_centroids

    plan = SequenceLengthPlan.from_host((97, 65), device)
    # integer block scores avoid near-tie ambiguity between reference and kernel reductions
    q = torch.ones(162, 2, 32, device=device, dtype=torch.bfloat16)
    k = (torch.arange(162, device=device) // 32 + 1).to(q.dtype)[:, None, None].expand_as(q).contiguous()
    layout, token, state = build_routes(q_rope=q, k_rope=k, length_plan=plan, chunk_size=32, topk=topk)
    _verify_against_host(layout)
    _check_route_invariants(token, state, layout)
    centroids = compute_block_centroids(k, layout)
    assert torch.equal(centroids, centroid_reference(k, layout.block_ptr_host))
    expected_token, expected_state, _ = route_reference(q, centroids, layout, topk - 1)
    assert torch.equal(token, expected_token)
    assert torch.equal(state, expected_state)
    metadata = build_direct_metadata(token, state, layout.Nc, retain_token_slots=True, derived_state_layout=layout)
    without_slots = build_direct_metadata(token, state, layout.Nc, derived_state_layout=layout)
    for actual, expected in ((without_slots.tok, metadata.tok), (without_slots.state, metadata.state)):
        if expected is None:
            assert actual is None
            continue
        assert actual.source_slots is None
        assert torch.equal(actual.count, expected.count)
        assert torch.equal(actual.offset, expected.offset)
        assert torch.equal(actual.rho, expected.rho)
    with pytest.raises(RuntimeError, match="Source slots require counts"):
        canonical_rho(state, layout.Nc, return_slots=True)
    for route, actual in ((token, metadata.tok), (state, metadata.state)):
        if route.shape[-1] == 0:
            assert actual is None
            continue
        count, offset, rows = metadata_reference(route, layout.Nc)
        assert torch.equal(actual.count.cpu(), count)
        assert torch.equal(actual.offset.cpu(), offset)
        assert _sort_rho_segments(actual) == rows
        assert torch.all(actual.rho[int(offset[-1]):] == -1)


@pytest.mark.parametrize("width", [8, 17, 136], ids=["below-minimum", "unaligned", "above-maximum"])
def test_head_dimension_rejected_before_launch(width, monkeypatch):
    from fla.ops.phba import chunk

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid geometry reached FlashAttention initialization")

    monkeypatch.setattr(chunk, "ensure_ready", unexpected)
    x = torch.zeros(1, 4, 1, width, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head dimensions"):
        chunk_phba(q=x, k=x, v=x, q_rope=x, k_rope=x, gate=x)


def test_length_plan_batch_boundaries():
    x = torch.zeros(2, 4, 1, 16, device=device, dtype=torch.bfloat16)
    values = torch.ones_like(x)
    values[0].fill_(10)
    wrong = SequenceLengthPlan.from_host([8], device)
    with pytest.raises(ValueError, match="dense batch boundary"):
        chunk_phba(q=x, k=x, v=values, q_rope=x, k_rope=x, gate=x, chunk_size=32, topk=1, length_plan=wrong)
    actual, _ = chunk_phba(q=x, k=x, v=values, q_rope=x, k_rope=x, gate=x, chunk_size=32, topk=1)
    assert torch.equal(actual, values)


def test_length_plan_cumulative_boundaries(monkeypatch):
    from fla.ops.phba.plans import _resolve_length_plan

    plan = SequenceLengthPlan.from_host((4, 4), device)
    wrong = torch.tensor([0, 8], device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="different boundaries"):
        _resolve_length_plan(1, 8, plan.device, wrong, plan)
    assert _resolve_length_plan(1, 8, plan.device, plan.cu_seqlens.clone(), plan) is plan

    def unexpected(*args, **kwargs):
        pytest.fail("Reusing the owned length tensor must not compare device values")

    monkeypatch.setattr(torch, "equal", unexpected)
    assert _resolve_length_plan(1, 8, plan.device, plan.cu_seqlens, plan) is plan


def test_low_memory_fallback_gradients(monkeypatch):
    from fla.ops.phba import plans

    torch.manual_seed(42)
    inputs = [torch.randn(1, 129, 2, 32, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(6)]
    do = torch.randn_like(inputs[0])

    def unexpected(*args, **kwargs):
        pytest.fail("Backward must use the tape plan saved by forward")

    def evaluate(grad=do):
        leaves = [x.clone().requires_grad_() for x in inputs]
        output, _ = chunk_phba(*leaves, chunk_size=32, topk=4)
        with monkeypatch.context() as context:
            context.setattr(plans, "owner_branch_tape_plan", unexpected)
            return output.detach(), torch.autograd.grad(output, leaves, grad)

    from fla.ops.phba.utils import prepare_grad_output

    unaligned = torch.empty(do.numel() + 1, dtype=do.dtype, device=do.device)[1:].view_as(do)
    unaligned.copy_(do)
    assert unaligned.is_contiguous() and unaligned.data_ptr() % 16 != 0
    assert prepare_grad_output(do, inputs[0]) is do
    aligned = prepare_grad_output(unaligned, inputs[0])
    assert aligned.data_ptr() % 16 == 0 and torch.equal(aligned, do)
    expected, expected_grads = evaluate()
    from fla.ops.phba import parallel

    history_run = parallel._history_partial.run

    def wide_history(*args, **kwargs):
        kwargs['INDEX32'] = False
        return history_run(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(parallel._history_partial, 'run', wide_history)
        wide_output, wide_grads = evaluate()
    assert torch.equal(wide_output, expected)
    for result, reference in zip(wide_grads, expected_grads):
        assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))
    offset_output, offset_grads = evaluate(unaligned)
    assert torch.equal(offset_output, expected)
    for result, reference in zip(offset_grads, expected_grads):
        assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))
    monkeypatch.setattr(plans, "OWNER_TAPE_CAP_BYTES", 0)
    monkeypatch.setattr(plans, "PREFIX_REVERSE_WORKSPACE_CAP", 0)
    assert not plans.prefix_backward_plan(5, 2, 32, 32).parallel
    assert not plans.owner_branch_tape_plan(129, 2, 5, 32, 32, 3, 4, 32, 2).enabled
    actual, grads = evaluate(unaligned)
    repeated, repeated_grads = evaluate()
    assert torch.equal(actual, repeated)
    assert_close("fallback_output", expected.float(), actual.float(), 0.03)
    for index, (reference, result, repeat) in enumerate(zip(expected_grads, grads, repeated_grads)):
        assert torch.isfinite(result).all()
        assert torch.equal(result.view(torch.uint8), repeat.view(torch.uint8))
        assert_close(f"fallback_gradient_{index}", reference.float(), result.float(), 0.06)


def test_length_plan_source_identity_and_mutation(monkeypatch):
    from fla.ops.phba.plans import _resolve_length_plan

    cu = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    plan = SequenceLengthPlan.from_device(cu)
    with monkeypatch.context() as context:

        def unexpected(*args, **kwargs):
            pytest.fail("Unchanged source lengths must not require a device comparison")

        context.setattr(torch, "equal", unexpected)
        assert _resolve_length_plan(1, 8, plan.device, cu, plan) is plan
    cu[1] = 3
    with pytest.raises(ValueError, match="different boundaries"):
        _resolve_length_plan(1, 8, plan.device, cu, plan)


def _sort_rho_segments(meta) -> list[list[int]]:
    """Read and sort each expert segment for metadata comparisons."""
    off = meta.offset.tolist()
    rho = meta.rho[: off[-1]].tolist()
    return [sorted(rho[off[e]: off[e + 1]]) for e in range(len(off) - 1)]


def _verify_against_host(layout) -> None:
    """Compare device block boundaries and sequence ownership with the host layout."""
    names = (
        "block_ptr",
        "seq_block_ptr",
        "block_seq",
        "current_block",
        "prefix_valid",
        "work_block",
        "work_start",
        "work_end",
        "work_seq",
    )
    t = {name: getattr(layout, name).tolist() for name in names}
    assert tuple(t["block_ptr"]) == layout.block_ptr_host
    assert tuple(t["seq_block_ptr"]) == layout.seq_block_ptr_host
    sbp = t["seq_block_ptr"]
    for b, s in enumerate(t["block_seq"]):
        assert sbp[s] <= b < sbp[s + 1]
    assert len(t["work_block"]) == layout.Nwork and len(t["current_block"]) == layout.T


def _check_route_invariants(rtok: torch.Tensor, rstate: torch.Tensor, layout) -> None:
    """Check route uniqueness, sequence boundaries, strict history and prefix validity."""
    H, T, kh = rtok.shape
    dev = rtok.device
    cb = layout.current_block.to(dev).long()
    bs = layout.block_seq.to(dev).long()
    r = rtok.long()
    valid = r >= 0
    assert not (bool((r[valid] >= layout.Nc).any()) or bool((r < -1).any())), "Token routes contain out-of-range block IDs"
    safe = r.clamp(min=0)
    assert not bool((valid & (bs[safe] != bs[cb][None, :, None])).any()), "Token routes cross sequence boundaries"
    assert not bool((valid & (safe >= cb[None, :, None])).any()), "Token routes contain nonhistorical blocks"
    if kh > 1:
        s = torch.sort(torch.where(valid, r, torch.arange(-kh, 0, device=dev).expand_as(r)), -1).values
        assert not bool((s[..., 1:] == s[..., :-1]).any()), "Token routes contain duplicate blocks in a row"
    expect = rstate_from_rtok(rtok, layout)
    assert torch.equal(rstate, expect), "State routes do not match the current/token routes and prefix validity"


def _reference_attention(q, k, v, cu, scale, causal):
    """Compute FP32 output and log-normalizers independently for each packed sequence."""
    T, H, D = q.shape
    out = torch.empty(T, H, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(H, T, dtype=torch.float32, device=q.device)
    cu = cu.tolist()
    for s in range(len(cu) - 1):
        a, b = cu[s], cu[s + 1]
        qs, ks, vs = (x[a:b].float().transpose(0, 1) for x in (q, k, v))  # [H, L, D]
        scores = qs @ ks.transpose(1, 2) * scale
        if causal:
            L = b - a
            mask = torch.ones(L, L, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        lse[:, a:b] = torch.logsumexp(scores, dim=-1)
        out[a:b] = (torch.softmax(scores, dim=-1) @ vs).transpose(0, 1)
    return out, lse


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dtype", "head_dim"),
    [
        pytest.param(torch.float16, 16, id="fp16-minimum-width"),
        pytest.param(torch.bfloat16, 64, id="bf16-aligned-width"),
        pytest.param(torch.float16, 72, id="fp16-nonpower-width"),
        pytest.param(torch.bfloat16, 128, id="bf16-maximum-width"),
    ],
)
def test_flash_attention_contract(dtype, head_dim):
    """Verify the packed low-level ABI and independent output/gradient reference."""
    torch.manual_seed(42)
    flash_attention.ensure_ready(device, dtype, head_dim)
    lengths, heads = (5, 17), 2
    total = sum(lengths)
    cu = torch.tensor((0, 5, total), dtype=torch.int32, device=device)
    q, k, v, do = [torch.randn(total, heads, head_dim, device=device, dtype=dtype) for _ in range(4)]
    scale = head_dim**-0.5
    with highest_precision():
        out, lse = flash_attention.varlen_forward(
            q,
            k,
            v,
            cu,
            cu,
            max(lengths),
            max(lengths),
            softmax_scale=scale,
            causal=True,
        )
        assert out.shape == q.shape and out.dtype == dtype
        assert lse.dtype == torch.float32 and lse.shape == (heads, total) and lse.is_contiguous()
        leaves = [x.detach().float().requires_grad_() for x in (q, k, v)]
        ref_out, ref_lse = _reference_attention(*leaves, cu, scale, causal=True)
        tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
        assert (out.float() - ref_out.detach()).abs().max() <= tolerance
        assert (lse - ref_lse.detach()).abs().max() <= 1e-2
        actual = [torch.empty_like(x) for x in (q, k, v)]
        # joint output/statistics must also work in independently allocated storage
        softmax_d = flash_attention.varlen_backward(
            do,
            q,
            k,
            v,
            out.clone(),
            lse.clone(),
            *actual,
            cu,
            cu,
            max(lengths),
            max(lengths),
            softmax_scale=scale,
            causal=True,
            deterministic=True,
        )
        assert isinstance(softmax_d, torch.Tensor)
        assert softmax_d.shape == (heads, total + 128 * len(lengths))
        reference = torch.autograd.grad(ref_out, leaves, do.float())
    for result, expected in zip(actual, reference):
        assert torch.isfinite(result).all()
        relative_max = (result.float() - expected).abs().max() / (expected.abs().max() + 1e-6)
        assert relative_max <= 5e-2


def test_flash_attention_preflight(monkeypatch):
    """Keep cached dependency/device/schema checks free of numerical self-tests."""

    def unexpected(*args, **kwargs):
        pytest.fail("Production preflight must not allocate probe tensors or run autograd")

    monkeypatch.setattr(flash_attention, "_READY", set())
    monkeypatch.setattr(flash_attention, "_SCHEMA_OK", False)
    monkeypatch.setattr(torch, "randn", unexpected)
    monkeypatch.setattr(torch.autograd, "grad", unexpected)
    monkeypatch.setattr(torch.Tensor, "backward", unexpected)
    with torch.inference_mode():
        flash_attention.ensure_ready(device, torch.bfloat16, 64)
        flash_attention.ensure_ready(device, torch.bfloat16, 64)
    assert len(flash_attention._READY) == 1
    with pytest.raises(RuntimeError, match="requires CUDA"):
        flash_attention.ensure_ready(torch.device("cpu"), torch.bfloat16, 64)
    with pytest.raises(RuntimeError, match="FP16/BF16"):
        flash_attention.ensure_ready(device, torch.float32, 64)
    with pytest.raises(RuntimeError, match="head dimension"):
        flash_attention.ensure_ready(device, torch.bfloat16, 8)

    def incompatible(q):
        pass

    check_schema = flash_attention.check_schema
    monkeypatch.setattr(flash_attention, "_READY", set())
    monkeypatch.setattr(flash_attention, "_SCHEMA_OK", False)
    monkeypatch.setattr(
        flash_attention,
        "check_schema",
        lambda: check_schema(forward_fn=incompatible, backward_fn=incompatible),
    )
    with pytest.raises(RuntimeError, match="Unsupported FA varlen"):
        flash_attention.ensure_ready(device, torch.bfloat16, 64)
    assert not flash_attention._READY and not flash_attention._SCHEMA_OK


def test_metadata_large_head_stride():
    """Count and sort a legal strided route whose final head offset exceeds signed int32."""
    from fla.ops.phba.metadata import build_direct_metadata
    from fla.ops.phba.naive import metadata_reference

    stride = 1 << 30
    required_bytes = (2 * stride + 33) * torch.empty((), dtype=torch.int32).element_size()
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_bytes + 1024**3:
        pytest.skip("Large-stride address coverage requires an 8GiB route allocation plus workspace")
    state = torch.empty_strided((3, 33, 1), (stride, 1, 1), dtype=torch.int32, device=device)
    state.fill_(-1)
    state[:, -1, :] = 1
    assert 2 * state.stride(0) > torch.iinfo(torch.int32).max
    token = torch.empty(3, 33, 0, dtype=torch.int32, device=device)
    result = build_direct_metadata(token, state, 2, derived_state_layout=None)
    expected_state = torch.full((3, 33, 1), -1, dtype=torch.int32, device="cpu")
    expected_state[:, -1, :] = 1
    expected_count, expected_offset, expected_rows = metadata_reference(expected_state, 2)
    assert result.tok is None and result.state.source_slots is None
    assert torch.equal(result.state.count.cpu(), expected_count)
    assert torch.equal(result.state.offset.cpu(), expected_offset)
    assert _sort_rho_segments(result.state) == expected_rows
    assert torch.all(result.state.rho[int(expected_offset[-1]):] == -1)
