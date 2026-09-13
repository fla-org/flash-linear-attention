# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend NPUGraph capture/replay coverage for KDA."""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from fla.ops.cp import FLACPContext
from fla.ops.kda import chunk_kda
from fla.utils import IS_NPU, device

pytestmark = pytest.mark.skipif(not IS_NPU, reason="KDA NPUGraph tests require an Ascend NPU")

if IS_NPU:
    assert hasattr(torch, "npu") and hasattr(torch.npu, "make_graphed_callables"), (
        "The Ascend graph test job requires torch.npu.make_graphed_callables"
    )

T = 128
H = 2
D = 64
N = 2
DTYPE = torch.float16
_INPUT_NAMES = ("q", "k", "v", "g", "beta", "h0")
_OUTPUT_NAMES = ("o", "ht", "dq", "dk", "dv", "dg", "db", "dh0")


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


def _make_inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device).manual_seed(seed)
    inputs = {
        "q": torch.randn(1, T, H, D, dtype=DTYPE, device=device, generator=generator),
        "k": F.normalize(
            torch.randn(1, T, H, D, dtype=torch.float32, device=device, generator=generator),
            dim=-1,
        ).to(DTYPE),
        "v": torch.randn(1, T, H, D, dtype=DTYPE, device=device, generator=generator),
        "g": F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float32, device=device, generator=generator)),
        "beta": torch.rand(1, T, H, dtype=DTYPE, device=device, generator=generator),
        "h0": torch.randn(N, H, D, D, dtype=torch.float32, device=device, generator=generator),
    }
    return {name: tensor.requires_grad_() for name, tensor in inputs.items()}


def _kda_graphable(q, k, v, g, beta, h0, cu_seqlens):
    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_graph=True,
        max_num_seqs=N,
    )


def _run_eager(inputs, cu_seqlens, do, dht, actual_t):
    leaves = {
        name: (
            inputs[name][:, :actual_t].detach().clone().requires_grad_()
            if name != "h0"
            else inputs[name].detach().clone().requires_grad_()
        )
        for name in _INPUT_NAMES
    }
    o, ht = chunk_kda(
        q=leaves["q"],
        k=leaves["k"],
        v=leaves["v"],
        g=leaves["g"],
        beta=leaves["beta"],
        initial_state=leaves["h0"],
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((o * do[:, :actual_t]).sum() + (ht * dht).sum()).backward()
    return [o.detach(), ht.detach(), *(leaves[name].grad for name in _INPUT_NAMES)]


def _make_case_inputs(case: _GraphCase, seed: int, requires_grad: bool = True) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device).manual_seed(seed)
    q = torch.randn(1, case.T, case.H, case.K, dtype=case.dtype, device=device, generator=generator)
    k = torch.randn(1, case.T, case.H, case.K, dtype=torch.float32, device=device, generator=generator)
    if not case.use_qk_l2norm_in_kernel:
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
    h0 = torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator) * 0.01
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


def _assert_training_case(case: _GraphCase, offsets: list[int]) -> None:
    sample = _make_case_inputs(case, seed=0)
    sample_offsets = [index * case.T // case.N for index in range(case.N)] + [case.T]
    sample_cu = torch.tensor(sample_offsets, dtype=torch.int64, device=device)

    def graph_step(q, k, v, g, beta, h0, A_log, dt_bias, cu_seqlens):
        return _call_case(case, (q, k, v, g, beta, h0, A_log, dt_bias), cu_seqlens, use_graph=True)

    graphed_step = torch.npu.make_graphed_callables(
        graph_step,
        sample + (sample_cu,),
        allow_unused_input=True,
    )
    inputs = _make_case_inputs(case, seed=1)
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)
    generator = torch.Generator(device).manual_seed(2)
    do = torch.randn(1, case.T, case.HV, case.V, dtype=case.dtype, device=device, generator=generator)
    state_shape = (case.N, case.HV, case.V, case.K) if case.state_v_first else (case.N, case.HV, case.K, case.V)
    dht = torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator)

    o, ht = graphed_step(*inputs, cu_seqlens)
    ((o * do).sum() + (ht * dht).sum()).backward()
    torch.npu.synchronize()

    actual_t = offsets[-1]
    cropped_reference_inputs = tuple(
        tensor[:, :actual_t].detach().clone().requires_grad_() if index < 5 else tensor
        for index, tensor in enumerate(reference_inputs)
    )
    reference_o, reference_ht = _call_case(case, cropped_reference_inputs, cu_seqlens, use_graph=False)
    ((reference_o * do[:, :actual_t]).sum() + (reference_ht * dht).sum()).backward()
    torch.npu.synchronize()

    torch.testing.assert_close(o[:, :actual_t], reference_o, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(ht, reference_ht, rtol=3e-3, atol=3e-3)
    names = ("q", "k", "v", "g", "beta", "h0", "A_log", "dt_bias")
    active_names = _active_input_names(case)
    for index, (name, actual, reference) in enumerate(zip(names, inputs, cropped_reference_inputs)):
        if name not in active_names:
            assert actual.grad is None
            continue
        actual_grad = actual.grad[:, :actual_t] if index < 5 else actual.grad
        torch.testing.assert_close(actual_grad, reference.grad, rtol=3e-3, atol=3e-3, msg=lambda m: f"{name}: {m}")

    if actual_t < case.T:
        for name, tensor in zip(("o", "dq", "dk", "dv", "dg", "db"), (o, *(inputs[i].grad for i in range(5)))):
            torch.testing.assert_close(
                tensor[:, actual_t:],
                torch.zeros_like(tensor[:, actual_t:]),
                rtol=0,
                atol=0,
                msg=lambda m, name=name: f"{name} padding: {m}",
            )


@pytest.mark.parametrize("cu_dtype", [torch.int32, torch.int64])
def test_chunk_kda_npugraph_multi_replay_matches_eager(cu_dtype):
    sample = _make_inputs(seed=0)
    sample_cu = torch.tensor([0, 1, T], dtype=cu_dtype, device=device)
    graphed_kda = torch.npu.make_graphed_callables(
        _kda_graphable,
        tuple(sample.values()) + (sample_cu,),
        allow_unused_input=True,
    )

    configs = (
        ("full-a", [0, 1, T]),
        ("full-b", [0, 65, T]),
        ("partial-zero-tail", [0, 64, 64]),
    )
    for seed, (tag, offsets) in enumerate(configs, start=1):
        inputs = _make_inputs(seed=seed)
        cu_seqlens = torch.tensor(offsets, dtype=cu_dtype, device=device)
        generator = torch.Generator(device).manual_seed(seed + 100)
        do = torch.randn(1, T, H, D, dtype=DTYPE, device=device, generator=generator)
        dht = torch.randn(N, H, D, D, dtype=torch.float32, device=device, generator=generator)

        o, ht = graphed_kda(*inputs.values(), cu_seqlens)
        ((o * do).sum() + (ht * dht).sum()).backward()
        torch.npu.synchronize()
        actual_t = offsets[-1]
        got = [o.detach().clone(), ht.detach().clone(), *(inputs[name].grad.detach().clone() for name in _INPUT_NAMES)]
        torch.npu.synchronize()
        expected = _run_eager(inputs, cu_seqlens, do, dht, actual_t)
        torch.npu.synchronize()

        for name, actual, reference in zip(_OUTPUT_NAMES, got, expected):
            if actual.ndim >= 2 and actual.shape[1] == T:
                actual = actual[:, :actual_t]
            torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3, msg=lambda m: f"{tag}::{name}: {m}")

        if actual_t < T:
            for name, tensor in zip(("o", "dq", "dk", "dv", "dg", "db"), (got[0], *got[2:7])):
                torch.testing.assert_close(
                    tensor[:, actual_t:],
                    torch.zeros_like(tensor[:, actual_t:]),
                    rtol=0,
                    atol=0,
                    msg=lambda m, name=name: f"{tag}::{name} padding: {m}",
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
            [0, 64, 64],
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
    ],
)
def test_chunk_kda_npugraph_option_matrix_matches_eager(case, offsets):
    _assert_training_case(case, offsets)


def test_chunk_kda_npugraph_without_final_state_matches_eager():
    case = _GraphCase(T=96, V=32, N=3, chunk_size=32, output_final_state=False)
    sample = _make_case_inputs(case, seed=0)
    sample_cu = torch.tensor([0, 32, 64, 96], dtype=torch.int32, device=device)

    def graph_step(q, k, v, g, beta, h0, A_log, dt_bias, cu_seqlens):
        return _call_case(case, (q, k, v, g, beta, h0, A_log, dt_bias), cu_seqlens, use_graph=True)[0]

    graphed_step = torch.npu.make_graphed_callables(
        graph_step,
        sample + (sample_cu,),
        allow_unused_input=True,
    )
    inputs = _make_case_inputs(case, seed=1)
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    cu_seqlens = torch.tensor([0, 31, 64, 96], dtype=torch.int32, device=device)
    generator = torch.Generator(device).manual_seed(2)
    do = torch.randn(1, case.T, case.HV, case.V, dtype=case.dtype, device=device, generator=generator)

    o = graphed_step(*inputs, cu_seqlens)
    (o * do).sum().backward()
    reference_o = _call_case(case, reference_inputs, cu_seqlens, use_graph=False)[0]
    (reference_o * do).sum().backward()
    torch.npu.synchronize()

    torch.testing.assert_close(o, reference_o, rtol=3e-3, atol=3e-3)
    for name, actual, reference in zip(("q", "k", "v", "g", "beta", "h0"), inputs[:6], reference_inputs[:6]):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=3e-3, atol=3e-3, msg=lambda m: f"{name}: {m}")


def test_chunk_kda_npugraph_intermediate_states_match_eager():
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
        graphed_step = torch.npu.make_graphed_callables(
            graph_step,
            sample + (sample_cu,),
            allow_unused_input=True,
        )
        inputs = _make_case_inputs(case, seed=1, requires_grad=False)
        cu_seqlens = torch.tensor([0, 64, 64, 64], dtype=torch.int64, device=device)
        o, ht, h = graphed_step(*inputs, cu_seqlens)
        reference_o, reference_ht, reference_h = _call_case(
            case,
            tuple(tensor[:, :64] if index < 5 else tensor for index, tensor in enumerate(inputs)),
            cu_seqlens,
            use_graph=False,
        )
        torch.npu.synchronize()

    torch.testing.assert_close(o[:, :64], reference_o, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(ht, reference_ht, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(h[:, :reference_h.shape[1]], reference_h, rtol=3e-3, atol=3e-3)
    assert h.shape[1] == (case.T + case.chunk_size - 1) // case.chunk_size + case.N - 1


def test_chunk_kda_npugraph_rejects_mismatched_sequence_capacity():
    inputs = _make_inputs(seed=0)
    cu_seqlens = torch.tensor([0, 64, T], dtype=torch.int64, device=device)

    with pytest.raises(ValueError, match=r"max_num_seqs \+ 1 entries"):
        chunk_kda(
            *(inputs[name] for name in ("q", "k", "v", "g", "beta")),
            initial_state=inputs["h0"],
            cu_seqlens=cu_seqlens,
            use_graph=True,
            max_num_seqs=N + 1,
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
def test_chunk_kda_npugraph_rejects_unsupported_options(kwargs, message):
    inputs = _make_inputs(seed=0)
    cu_seqlens = torch.tensor([0, 64, T], dtype=torch.int64, device=device)
    kwargs.setdefault("cu_seqlens", cu_seqlens)

    with pytest.raises(NotImplementedError, match=message):
        chunk_kda(
            *(inputs[name] for name in ("q", "k", "v", "g", "beta")),
            initial_state=inputs["h0"],
            use_graph=True,
            max_num_seqs=N,
            **kwargs,
        )
