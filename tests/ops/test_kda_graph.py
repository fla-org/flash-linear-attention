# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Graph capture/replay tests for KDA chunk training (fwd + bwd).

Captures a fwd+bwd step once, replays it with different ``cu_seqlens`` contents
(different splits, zero-length tail padding, and total_tokens below the static
capacity), and compares against the eager (``use_graph=False``) path.
Covers CUDA direct capture and NPU graphed callables.
"""

import os
from dataclasses import dataclass
from importlib import import_module

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.ops.cp import FLACPContext, build_cp_context
from fla.ops.kda import chunk_kda
from fla.utils import IS_AMD, IS_NPU, IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(
    not (IS_NVIDIA or IS_AMD or IS_NPU),
    reason="KDA graph tests require CUDA/HIP or Ascend NPU",
)
cuda_only = pytest.mark.skipif(IS_NPU, reason="direct CUDA capture and NCCL require a CUDA/HIP device")
npu_only = pytest.mark.skipif(not IS_NPU, reason="graphed callable coverage is currently validated on Ascend NPU")
graph_device = torch.npu if IS_NPU else torch.cuda


# static capture capacity
T = 1024
H = 4
D = 64
MAX_NUM_SEQS = 4

# (tag, cu_seqlens) -- all cover [0, T]; trailing repeats are zero-length padding sequences
_VARLEN_CONFIGS = [
    ("4seq", [0, 300, 600, 900, 1024]),
    ("2seq_zerotail", [0, 512, 1024, 1024, 1024]),
    ("4seq_alt", [0, 100, 400, 700, 1024]),
    ("1seq_zerotail", [0, 1024, 1024, 1024, 1024]),
]

# (tag, cu_seqlens) -- total_tokens < T; rows above cu[-1] are padding and must stay inert
_PARTIAL_CONFIGS = [
    ("half", [0, 512, 512, 512, 512]),
    ("quarter_2seq", [0, 128, 384, 384, 384]),
]

_OUT_NAMES = ["o", "ht", "dq", "dk", "dv", "dg", "db", "dh0", "dA", "dbias"]


def _rand_inputs(seed, gate, safe_gate, hv=H):
    # q/k carry H heads; v/g/beta/state carry HV heads (HV>H exercises GVA).
    g = torch.Generator(device).manual_seed(seed)
    dt = torch.bfloat16
    q = torch.randn(1, T, H, D, dtype=dt, device=device, generator=g)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32, device=device, generator=g), p=2, dim=-1).to(dt)
    v = torch.randn(1, T, hv, D, dtype=dt, device=device, generator=g)
    if gate:
        gg = torch.randn(1, T, hv, D, dtype=dt, device=device, generator=g)
        A_log = torch.log(torch.empty(1, 1, hv, 1, device=device).uniform_(1, 16, generator=g))
        dt_bias = torch.randn(hv * D, dtype=torch.float32, device=device, generator=g)
    else:
        gg = F.logsigmoid(torch.randn(1, T, hv, D, dtype=torch.float32, device=device, generator=g))
        if safe_gate:
            gg = gg.clamp(-5, 0)
        A_log = dt_bias = None
    beta = torch.rand(1, T, hv, dtype=dt, device=device, generator=g).sigmoid()
    h0 = torch.randn(MAX_NUM_SEQS, hv, D, D, dtype=torch.float32, device=device, generator=g)
    do = torch.randn(1, T, hv, D, dtype=dt, device=device, generator=g)
    dht = torch.randn(MAX_NUM_SEQS, hv, D, D, dtype=torch.float32, device=device, generator=g)
    return dict(q=q, k=k, v=v, g=gg, beta=beta, h0=h0, A_log=A_log, dt_bias=dt_bias, do=do, dht=dht)


def _eager(inp, cu, gate, safe_gate):
    leaves = {n: inp[n].detach().clone().requires_grad_() for n in ("q", "k", "v", "g", "beta", "h0")}
    extra = {}
    if gate:
        leaves["A_log"] = inp["A_log"].detach().clone().requires_grad_()
        leaves["dt_bias"] = inp["dt_bias"].detach().clone().requires_grad_()
        extra = dict(A_log=leaves["A_log"], dt_bias=leaves["dt_bias"])
    o, ht = chunk_kda(
        q=leaves["q"],
        k=leaves["k"],
        v=leaves["v"],
        g=leaves["g"],
        beta=leaves["beta"],
        initial_state=leaves["h0"],
        output_final_state=True,
        cu_seqlens=cu,
        use_gate_in_kernel=gate,
        safe_gate=safe_gate,
        lower_bound=(-5 if safe_gate else None),
        **extra,
    )
    ((o * inp["do"]).sum() + (ht * inp["dht"]).sum()).backward()
    out = [
        o.detach(),
        ht.detach(),
        leaves["q"].grad,
        leaves["k"].grad,
        leaves["v"].grad,
        leaves["g"].grad,
        leaves["beta"].grad,
        leaves["h0"].grad,
    ]
    out += [leaves["A_log"].grad, leaves["dt_bias"].grad] if gate else [None, None]
    return out


def _make_graphed(inp, cu, gate, safe_gate):
    """Build static leaf buffers and capture a fwd+bwd step. Returns (graph, leaves, do, dht, cap_o, cap_ht)."""
    leaves = {n: inp[n].detach().clone().requires_grad_() for n in ("q", "k", "v", "g", "beta", "h0")}
    extra = {}
    if gate:
        leaves["A_log"] = inp["A_log"].detach().clone().requires_grad_()
        leaves["dt_bias"] = inp["dt_bias"].detach().clone().requires_grad_()
        extra = dict(A_log=leaves["A_log"], dt_bias=leaves["dt_bias"])
    do = inp["do"].detach().clone()
    dht = inp["dht"].detach().clone()
    grad_leaves = [leaves[n] for n in leaves]

    def step():
        for t in grad_leaves:
            if t.grad is not None:
                t.grad.zero_()
        o, ht = chunk_kda(
            q=leaves["q"],
            k=leaves["k"],
            v=leaves["v"],
            g=leaves["g"],
            beta=leaves["beta"],
            initial_state=leaves["h0"],
            output_final_state=True,
            cu_seqlens=cu,
            use_gate_in_kernel=gate,
            safe_gate=safe_gate,
            lower_bound=(-5 if safe_gate else None),
            use_graph=True,
            max_num_seqs=MAX_NUM_SEQS,
            **extra,
        )
        ((o * do).sum() + (ht * dht).sum()).backward()
        return o.detach().clone(), ht.detach().clone()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cap_o, cap_ht = step()
    return graph, leaves, do, dht, cap_o, cap_ht


def _graphed_outputs(leaves, gate, cap_o, cap_ht):
    out = [
        cap_o,
        cap_ht,
        leaves["q"].grad,
        leaves["k"].grad,
        leaves["v"].grad,
        leaves["g"].grad,
        leaves["beta"].grad,
        leaves["h0"].grad,
    ]
    out += [leaves["A_log"].grad, leaves["dt_bias"].grad] if gate else [None, None]
    return out


# (gate, safe_gate, hv) covering every chunk_kda config axis incl. GVA (hv > H).
_CFG = [
    pytest.param(False, False, H, id="default"),
    pytest.param(False, True, H, id="safe_gate"),
    pytest.param(True, False, H, id="gate_in_kernel"),
    pytest.param(True, True, H, id="gate+safe_gate"),
    pytest.param(False, False, 2 * H, id="gva_default"),
    pytest.param(True, True, 2 * H, id="gva_gate+safe"),
]


@pytest.mark.parametrize(("gate", "safe_gate", "hv"), _CFG)
@cuda_only
def test_chunk_kda_graph_replay_matches_eager(gate, safe_gate, hv):
    """Capture fwd+bwd, replay with the captured inputs, and compare to the eager path."""
    cu = torch.tensor(_VARLEN_CONFIGS[0][1], dtype=torch.long, device=device)
    inp = _rand_inputs(seed=0, gate=gate, safe_gate=safe_gate, hv=hv)

    graph, leaves, _do, _dht, cap_o, cap_ht = _make_graphed(inp, cu, gate, safe_gate)
    graph.replay()
    torch.cuda.synchronize()

    got = _graphed_outputs(leaves, gate, cap_o, cap_ht)
    ref = _eager(inp, cu, gate, safe_gate)
    for name, r, g in zip(_OUT_NAMES, ref, got):
        if r is None:
            continue
        assert_close(f"graph::{name}", r, g, 2e-3)


@pytest.mark.parametrize(("gate", "safe_gate", "hv"), _CFG)
@cuda_only
def test_chunk_kda_graph_multi_replay_varlen(gate, safe_gate, hv):
    """Capture once, then replay against several different cu_seqlens layouts.

    Each replay copies fresh inputs + a new cu_seqlens content into the static buffers and
    must match an independent eager run on that same data -- proving the in-graph device-side
    chunk-index rebuild recomputes correctly from the live cu_seqlens on every replay.
    """
    cu = torch.tensor(_VARLEN_CONFIGS[0][1], dtype=torch.long, device=device)
    inp0 = _rand_inputs(seed=0, gate=gate, safe_gate=safe_gate, hv=hv)
    graph, leaves, _do, _dht, cap_o, cap_ht = _make_graphed(inp0, cu, gate, safe_gate)

    update_names = ["q", "k", "v", "g", "beta", "h0"] + (["A_log", "dt_bias"] if gate else [])
    for seed, (tag, cucfg) in enumerate(_VARLEN_CONFIGS, start=1):
        inp = _rand_inputs(seed=seed * 17, gate=gate, safe_gate=safe_gate, hv=hv)
        for n in update_names:
            leaves[n].data.copy_(inp[n])
        _do.copy_(inp["do"])
        _dht.copy_(inp["dht"])
        cu.copy_(torch.tensor(cucfg, dtype=torch.long, device=device))

        graph.replay()
        torch.cuda.synchronize()

        got = _graphed_outputs(leaves, gate, cap_o, cap_ht)
        ref = _eager(inp, torch.tensor(cucfg, dtype=torch.long, device=device), gate, safe_gate)
        for name, r, g in zip(_OUT_NAMES, ref, got):
            if r is None:
                continue
            assert_close(f"{tag}::{name}", r, g, 2e-3)


@pytest.mark.parametrize(("safe_gate", "hv"), [
    pytest.param(False, H, id="gate_in_kernel"),
    pytest.param(True, H, id="gate+safe_gate"),
    pytest.param(True, 2 * H, id="gva_gate+safe"),
])
@cuda_only
def test_chunk_kda_graph_partial_tokens(safe_gate, hv):
    """Replay with total_tokens < T_static: padding rows must not pollute dA/dbias.

    The capture config covers all T rows, so on a partial replay the padding rows
    of intermediate buffers hold stale nonzero values. Full-tensor reductions in
    kda_gate_bwd (dA/dbias) must still match the eager path run on real tokens only.
    """
    gate = True
    cu = torch.tensor(_VARLEN_CONFIGS[0][1], dtype=torch.long, device=device)
    inp0 = _rand_inputs(seed=0, gate=gate, safe_gate=safe_gate, hv=hv)
    graph, leaves, _do, _dht, cap_o, cap_ht = _make_graphed(inp0, cu, gate, safe_gate)

    update_names = ["q", "k", "v", "g", "beta", "h0", "A_log", "dt_bias"]
    for seed, (tag, cucfg) in enumerate(_PARTIAL_CONFIGS, start=1):
        n = cucfg[-1]
        inp = _rand_inputs(seed=seed * 17, gate=gate, safe_gate=safe_gate, hv=hv)
        for name in update_names:
            leaves[name].data.copy_(inp[name])
        _do.copy_(inp["do"])
        _dht.copy_(inp["dht"])
        cu.copy_(torch.tensor(cucfg, dtype=torch.long, device=device))

        graph.replay()
        torch.cuda.synchronize()

        got = _graphed_outputs(leaves, gate, cap_o, cap_ht)
        sliced = {k: v[:, :n] if k in ("q", "k", "v", "g", "beta", "do") else v for k, v in inp.items()}
        ref = _eager(sliced, torch.tensor(cucfg, dtype=torch.long, device=device), gate, safe_gate)
        for name, r, g in zip(_OUT_NAMES, ref, got):
            if r is None:
                continue
            if name in ("o", "dq", "dk", "dv", "dg", "db"):
                g = g[:, :n]
            assert_close(f"partial::{tag}::{name}", r, g, 2e-3)


def _eager_cp(inp, cucfg, gate):
    """Eager CP reference: world_size=1 crosses no rank, so only real tokens matter."""
    leaves = {n: inp[n].detach().clone().requires_grad_() for n in ("q", "k", "v", "g", "beta")}
    extra = {}
    if gate:
        leaves["A_log"] = inp["A_log"].detach().clone().requires_grad_()
        leaves["dt_bias"] = inp["dt_bias"].detach().clone().requires_grad_()
        extra = dict(A_log=leaves["A_log"], dt_bias=leaves["dt_bias"])
    cu = torch.tensor(cucfg, dtype=torch.long, device=device)
    o, _ = chunk_kda(
        q=leaves["q"],
        k=leaves["k"],
        v=leaves["v"],
        g=leaves["g"],
        beta=leaves["beta"],
        cp_context=build_cp_context(cu, group=dist.group.WORLD),
        use_gate_in_kernel=gate,
        **extra,
    )
    (o * inp["do"]).sum().backward()
    out = [o.detach(), leaves["q"].grad, leaves["k"].grad, leaves["v"].grad, leaves["g"].grad, leaves["beta"].grad]
    out += [leaves["A_log"].grad, leaves["dt_bias"].grad] if gate else [None, None]
    return out


@pytest.mark.skipif(not dist.is_available(), reason="requires torch.distributed")
@cuda_only
def test_chunk_kda_graph_cp_world1():
    """world_size=1 CP + graph: exercises the recorded all-gather and device-side CP metadata.

    A single rank is both first and last (pre/post_num_ranks = 0), so no state crosses
    ranks and the graph-mode CP branch must match the eager CP branch. The context is
    built once around the persistent buffers: rebuilding it per step would D2H-sync,
    which capture forbids. Multi-rank numerics are covered by tests/context_parallel.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29617")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    try:
        gate = True
        cu = torch.tensor(_VARLEN_CONFIGS[0][1], dtype=torch.long, device=device)
        pre_dev = torch.zeros(1, dtype=torch.int32, device=device)
        post_dev = torch.zeros(1, dtype=torch.int32, device=device)
        base = build_cp_context(cu, group=dist.group.WORLD)
        ctx = FLACPContext(
            group=dist.group.WORLD,
            cu_seqlens=cu,
            cu_seqlens_cpu=base.cu_seqlens_cpu,
            is_last_rank=base.is_last_rank,
            pre_num_ranks=base.pre_num_ranks,
            is_first_rank=base.is_first_rank,
            post_num_ranks=base.post_num_ranks,
            pre_num_ranks_dev=pre_dev,
            post_num_ranks_dev=post_dev,
        )

        inp0 = _rand_inputs(seed=0, gate=gate, safe_gate=False)
        names = ["q", "k", "v", "g", "beta", "A_log", "dt_bias"]
        leaves = {n: inp0[n].detach().clone().requires_grad_() for n in names}
        do = inp0["do"].detach().clone()

        def step():
            for t in leaves.values():
                if t.grad is not None:
                    t.grad.zero_()
            o, _ = chunk_kda(
                q=leaves["q"],
                k=leaves["k"],
                v=leaves["v"],
                g=leaves["g"],
                beta=leaves["beta"],
                cp_context=ctx,
                use_gate_in_kernel=gate,
                A_log=leaves["A_log"],
                dt_bias=leaves["dt_bias"],
                use_graph=True,
                max_num_seqs=MAX_NUM_SEQS,
            )
            (o * do).sum().backward()
            return o.detach().clone()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cap_o = step()

        names_out = ["o", "dq", "dk", "dv", "dg", "db", "dA", "dbias"]
        for seed, (tag, cucfg) in enumerate(_VARLEN_CONFIGS, start=1):
            inp = _rand_inputs(seed=seed * 17, gate=gate, safe_gate=False)
            for n in names:
                leaves[n].data.copy_(inp[n])
            do.copy_(inp["do"])
            cu.copy_(torch.tensor(cucfg, dtype=torch.long, device=device))

            graph.replay()
            torch.cuda.synchronize()

            got = [cap_o] + [leaves[n].grad for n in names]
            ref = _eager_cp(inp, cucfg, gate)
            for name, r, g in zip(names_out, ref, got):
                assert_close(f"cp_world1::{tag}::{name}", r, g, 2e-3)
    finally:
        dist.destroy_process_group()


@dataclass(frozen=True)
class _GraphCase:
    T: int = 128
    H: int = 2
    HV: int = 2
    K: int = 64
    V: int = 64
    N: int = 2
    dtype: torch.dtype = torch.float16
    chunk_size: int = 64
    use_qk_l2norm_in_kernel: bool = False
    use_gate_in_kernel: bool = False
    use_a_log: bool = True
    use_dt_bias: bool = True
    use_beta_sigmoid_in_kernel: bool = False
    allow_neg_eigval: bool = False
    safe_gate: bool = False
    disable_recompute: bool = False
    return_intermediate_states: bool = False
    state_v_first: bool = False
    use_initial_state: bool = True
    output_final_state: bool = True
    cu_dtype: torch.dtype = torch.int64
    normalize_q: bool = True
    initial_state_scale: float = 0.01


def _make_case_inputs(case: _GraphCase, seed: int, requires_grad: bool = True) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device).manual_seed(seed)
    q = torch.randn(1, case.T, case.H, case.K, dtype=case.dtype, device=device, generator=generator)
    k = torch.randn(1, case.T, case.H, case.K, dtype=torch.float32, device=device, generator=generator)
    if not case.use_qk_l2norm_in_kernel:
        if case.normalize_q:
            q = F.normalize(q.float(), dim=-1).to(case.dtype)
        k = F.normalize(k, dim=-1)
    k = k.to(case.dtype)
    v = torch.randn(1, case.T, case.HV, case.V, dtype=case.dtype, device=device, generator=generator)
    if case.use_gate_in_kernel:
        g = torch.randn(1, case.T, case.HV, case.K, dtype=case.dtype, device=device, generator=generator)
    else:
        g = F.logsigmoid(
            torch.randn(1, case.T, case.HV, case.K, dtype=torch.float32, device=device, generator=generator)
        )
    if case.use_beta_sigmoid_in_kernel:
        beta = torch.randn(1, case.T, case.HV, dtype=case.dtype, device=device, generator=generator)
    else:
        beta = torch.rand(1, case.T, case.HV, dtype=case.dtype, device=device, generator=generator)
    state_shape = (case.N, case.HV, case.V, case.K) if case.state_v_first else (case.N, case.HV, case.K, case.V)
    h0 = torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator) * case.initial_state_scale
    A_log = torch.randn(case.HV, dtype=torch.float32, device=device, generator=generator)
    dt_bias = torch.randn(case.HV * case.K, dtype=torch.float32, device=device, generator=generator)
    inputs = (q, k, v, g, beta, h0, A_log, dt_bias)
    return tuple(tensor.requires_grad_(requires_grad) for tensor in inputs)


def _call_case(case: _GraphCase, inputs: tuple[torch.Tensor, ...], cu_seqlens: torch.Tensor, use_graph: bool):
    q, k, v, g, beta, h0, A_log, dt_bias = inputs
    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0 if case.use_initial_state else None,
        output_final_state=case.output_final_state,
        use_qk_l2norm_in_kernel=case.use_qk_l2norm_in_kernel,
        use_gate_in_kernel=case.use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=case.use_beta_sigmoid_in_kernel,
        allow_neg_eigval=case.allow_neg_eigval,
        safe_gate=case.safe_gate,
        lower_bound=-5.0 if case.safe_gate else None,
        disable_recompute=case.disable_recompute,
        return_intermediate_states=case.return_intermediate_states,
        state_v_first=case.state_v_first,
        cu_seqlens=cu_seqlens,
        use_graph=use_graph,
        max_num_seqs=case.N,
        A_log=A_log if case.use_gate_in_kernel and case.use_a_log else None,
        dt_bias=dt_bias if case.use_gate_in_kernel and case.use_dt_bias else None,
        chunk_size=case.chunk_size,
    )


def _active_input_names(case: _GraphCase) -> tuple[str, ...]:
    names = ["q", "k", "v", "g", "beta"]
    if case.use_initial_state:
        names.append("h0")
    if case.use_gate_in_kernel:
        if case.use_a_log:
            names.append("A_log")
        if case.use_dt_bias:
            names.append("dt_bias")
    return tuple(names)


def _assert_training_case(case: _GraphCase, offsets: list[int] | tuple[list[int], ...], tolerance: float = 3e-3) -> None:
    sample = _make_case_inputs(case, seed=0)
    sample_offsets = [i * case.T // case.N for i in range(case.N)] + [case.T]
    sample_cu = torch.tensor(sample_offsets, dtype=case.cu_dtype, device=device)

    def graph_step(q, k, v, g, beta, h0, A_log, dt_bias, cu_seqlens):
        result = _call_case(case, (q, k, v, g, beta, h0, A_log, dt_bias), cu_seqlens, use_graph=True)
        return result if case.output_final_state else result[0]

    graphed_step = graph_device.make_graphed_callables(
        graph_step,
        sample + (sample_cu,),
        allow_unused_input=True,
    )
    replay_offsets = (offsets,) if isinstance(offsets[0], int) else offsets
    for replay, current_offsets in enumerate(replay_offsets, start=1):
        inputs = _make_case_inputs(case, seed=replay)
        reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
        cu_seqlens = torch.tensor(current_offsets, dtype=case.cu_dtype, device=device)
        generator = torch.Generator(device).manual_seed(replay + 100)
        do = torch.randn(1, case.T, case.HV, case.V, dtype=case.dtype, device=device, generator=generator)
        state_shape = (case.N, case.HV, case.V, case.K) if case.state_v_first else (
            case.N,
            case.HV,
            case.K,
            case.V,
        )
        dht = torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator)

        result = graphed_step(*inputs, cu_seqlens)
        o, ht = result if case.output_final_state else (result, None)
        loss = (o * do).sum()
        if ht is not None:
            loss = loss + (ht * dht).sum()
        loss.backward()
        graph_device.synchronize()

        actual_t = current_offsets[-1]
        cropped_reference_inputs = tuple(
            tensor[:, :actual_t].detach().clone().requires_grad_() if index < 5 else tensor
            for index, tensor in enumerate(reference_inputs)
        )
        reference_o, reference_ht = _call_case(case, cropped_reference_inputs, cu_seqlens, use_graph=False)
        reference_loss = (reference_o * do[:, :actual_t]).sum()
        if reference_ht is not None:
            reference_loss = reference_loss + (reference_ht * dht).sum()
        reference_loss.backward()
        graph_device.synchronize()

        torch.testing.assert_close(o[:, :actual_t], reference_o, rtol=tolerance, atol=tolerance)
        if case.output_final_state:
            torch.testing.assert_close(ht, reference_ht, rtol=tolerance, atol=tolerance)
        else:
            assert reference_ht is None
        names = ("q", "k", "v", "g", "beta", "h0", "A_log", "dt_bias")
        active_names = _active_input_names(case)
        for index, (name, actual, reference) in enumerate(zip(names, inputs, cropped_reference_inputs)):
            if name not in active_names:
                assert actual.grad is None
                continue
            actual_grad = actual.grad[:, :actual_t] if index < 5 else actual.grad
            assert torch.isfinite(actual_grad).all(), f"{current_offsets}::{name} graph gradient is not finite"
            assert torch.isfinite(reference.grad).all(), f"{current_offsets}::{name} eager gradient is not finite"
            torch.testing.assert_close(
                actual_grad,
                reference.grad,
                rtol=tolerance,
                atol=tolerance,
                msg=lambda m: f"{current_offsets}::{name}: {m}",
            )

        if IS_NPU and actual_t < case.T:
            for name, tensor in zip(("o", "dq", "dk", "dv", "dg", "db"), (o, *(inputs[i].grad for i in range(5)))):
                torch.testing.assert_close(
                    tensor[:, actual_t:],
                    torch.zeros_like(tensor[:, actual_t:]),
                    rtol=0,
                    atol=0,
                    msg=lambda m, name=name: f"{name} padding: {m}",
                )


@pytest.mark.parametrize("cu_dtype", [torch.int32, torch.int64])
@npu_only
def test_chunk_kda_graph_callable_multi_replay_matches_eager(cu_dtype):
    _assert_training_case(
        case=_GraphCase(cu_dtype=cu_dtype, normalize_q=False, initial_state_scale=1.0),
        offsets=([0, 1, 128], [0, 65, 128], [0, 64, 64], [0, 1, 128]),
        tolerance=2e-3,
    )


@npu_only
def test_chunk_kda_graph_callable_skips_sentinels_between_tasks(monkeypatch):
    """Invalid tasks must not prevent the same core from processing later valid tasks."""
    for name in (
        "fla.ops.kda.backends.triton_ascend.chunk_bwd",
        "fla.ops.kda.backends.triton_ascend.chunk_intra",
        "fla.ops.kda.backends.triton_ascend.wy_fast",
        "fla.ops.gla.backends.triton_ascend.chunk",
    ):
        module = import_module(name)
        properties = {**module.get_npu_properties(), "num_aicore": 1, "num_vectorcore": 1}
        monkeypatch.setattr(module, "get_npu_properties", lambda properties=properties: properties)
    _assert_training_case(
        case=_GraphCase(T=128, N=4),
        offsets=([0, 0, 64, 128, 128], [0, 64, 64, 64, 64], [0, 0, 64, 128, 128]),
    )


@pytest.mark.parametrize(
    ("case", "offsets"),
    [
        pytest.param(
            _GraphCase(
                T=96,
                H=1,
                HV=2,
                K=64,
                V=32,
                N=3,
                dtype=torch.bfloat16,
                chunk_size=32,
                use_qk_l2norm_in_kernel=True,
                use_initial_state=False,
            ),
            [0, 17, 96, 96],
            id="bf16-bt32-gva-k-ne-v-l2-no-h0",
        ),
        pytest.param(
            _GraphCase(
                V=32,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                disable_recompute=True,
            ),
            [0, 65, 128],
            id="fp16-fused-gate-beta-no-recompute",
        ),
        pytest.param(
            _GraphCase(
                T=96,
                dtype=torch.bfloat16,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=True,
                safe_gate=True,
                state_v_first=True,
            ),
            (
                [0, 1, 96],
                [0, 65, 96],
                [0, 64, 64],
                [0, 1, 96],
            ),
            id="bf16-production-safe-gate-partial",
        ),
        pytest.param(
            _GraphCase(
                T=64,
                H=1,
                HV=1,
                V=32,
                use_gate_in_kernel=True,
                use_a_log=False,
                use_dt_bias=False,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
            ),
            [0, 32, 64],
            id="fp16-safe-gate-without-a-or-bias",
        ),
        pytest.param(
            _GraphCase(T=96, V=32, N=3, chunk_size=32, output_final_state=False, cu_dtype=torch.int32),
            [0, 31, 64, 96],
            id="fp16-bt32-without-final-state",
        ),
        pytest.param(
            _GraphCase(T=2048, H=1, HV=1, K=128, V=128),
            ([0, 1023, 2048], [0, 65, 1024], [0, 1, 2048]),
            id="fp16-long-sequence-large-head",
        ),
        pytest.param(
            _GraphCase(T=128, K=100, V=100),
            ([0, 1, 128], [0, 64, 64], [0, 1, 128]),
            id="fp16-unaligned-head-partial",
        ),
        pytest.param(
            _GraphCase(T=4, N=8),
            [0, 0, 1, 1, 2, 2, 3, 3, 4],
            id="fp16-more-sequence-slots-than-tokens",
        ),
    ],
)
@npu_only
def test_chunk_kda_graph_callable_option_matrix_matches_eager(case, offsets):
    _assert_training_case(case, offsets)


@npu_only
def test_chunk_kda_graph_callable_intermediate_states_match_eager():
    case = _GraphCase(
        T=96,
        N=3,
        dtype=torch.bfloat16,
        return_intermediate_states=True,
    )
    sample = _make_case_inputs(case, seed=0, requires_grad=False)
    sample_cu = torch.tensor([0, 32, 64, 96], dtype=torch.int64, device=device)

    def graph_step(q, k, v, g, beta, h0, A_log, dt_bias, cu_seqlens):
        return _call_case(case, (q, k, v, g, beta, h0, A_log, dt_bias), cu_seqlens, use_graph=True)

    with torch.inference_mode():
        graphed_step = graph_device.make_graphed_callables(
            graph_step,
            sample + (sample_cu,),
            allow_unused_input=True,
        )
        for seed, offsets in enumerate(([0, 1, 2, 96], [0, 64, 64, 64], [0, 1, 2, 96]), start=1):
            inputs = _make_case_inputs(case, seed=seed, requires_grad=False)
            cu_seqlens = torch.tensor(offsets, dtype=case.cu_dtype, device=device)
            actual_t = offsets[-1]
            o, ht, h = graphed_step(*inputs, cu_seqlens)
            reference_o, reference_ht, reference_h = _call_case(
                case,
                tuple(tensor[:, :actual_t] if index < 5 else tensor for index, tensor in enumerate(inputs)),
                cu_seqlens,
                use_graph=False,
            )
            graph_device.synchronize()

            torch.testing.assert_close(o[:, :actual_t], reference_o, rtol=3e-3, atol=3e-3)
            torch.testing.assert_close(ht, reference_ht, rtol=3e-3, atol=3e-3)
            torch.testing.assert_close(h[:, :reference_h.shape[1]], reference_h, rtol=3e-3, atol=3e-3)
            assert h.shape[1] == (case.T + case.chunk_size - 1) // case.chunk_size + case.N - 1


@npu_only
def test_chunk_kda_graph_callable_rejects_mismatched_sequence_capacity():
    case = _GraphCase()
    inputs = _make_case_inputs(case, seed=0)
    cu_seqlens = torch.tensor([0, 64, case.T], dtype=case.cu_dtype, device=device)

    with pytest.raises(AssertionError, match=r"max_num_seqs \+ 1 entries"):
        chunk_kda(
            *inputs[:5],
            initial_state=inputs[5],
            cu_seqlens=cu_seqlens,
            use_graph=True,
            max_num_seqs=case.N + 1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"cp_context": FLACPContext()},
            "does not currently support context parallelism",
            id="context-parallel",
        ),
        pytest.param({"cu_seqlens": None}, "requires flattened variable-length inputs", id="dense"),
    ],
)
@pytest.mark.skipif(not IS_NPU, reason="Ascend-specific graph limitations")
def test_chunk_kda_graph_callable_rejects_unsupported_options(kwargs, message):
    case = _GraphCase()
    inputs = _make_case_inputs(case, seed=0)
    cu_seqlens = torch.tensor([0, 64, case.T], dtype=case.cu_dtype, device=device)
    kwargs.setdefault("cu_seqlens", cu_seqlens)

    with pytest.raises(NotImplementedError, match=message):
        chunk_kda(
            *inputs[:5],
            initial_state=inputs[5],
            use_graph=True,
            max_num_seqs=case.N,
            **kwargs,
        )
