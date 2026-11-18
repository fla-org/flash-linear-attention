# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
"""Numerical and routing coverage for the exported B300 forward backend."""

import inspect

import pytest
import torch
import torch.nn.functional as F

from fla.backends import BaseBackend, _function_registries
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.backends.qla_ptx import WORKLOADS, QLAPTXBackend, _runtime, chunk_async
from fla.utils import assert_close, device, get_device_capability

pytestmark = pytest.mark.skipif(device != "cuda" or get_device_capability() != (10, 3), reason="requires NVIDIA SM103")


def inputs(name, has_state):
    torch.manual_seed(42)
    w = next(w for w in WORKLOADS if w["name"] == name)
    b, t, hq, hv = (w[k] for k in ("B", "T", "Hq", "Hv"))
    q, k = [F.normalize(torch.randn(b, t, hq, 128, device=device), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(b, t, hv, 128, device=device, dtype=torch.bfloat16)
    a = torch.rand(hv, device=device) * 16
    a[0], a[-1] = 0, 16
    g = -a * F.softplus(torch.randn(b, t, hv, device=device) * 0.5 + 1)
    beta = torch.randn(b, t, hv, device=device).sigmoid()
    cpu_cu = torch.tensor(w["cu"], dtype=torch.int64) if "cu" in w else None
    n = b if cpu_cu is None else len(cpu_cu) - 1
    state = torch.randn(n, hv, 128, 128, device=device) if has_state else None
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        cu_seqlens=cpu_cu.to(device) if cpu_cu is not None else None,
        cu_seqlens_cpu=cpu_cu,
    )


@pytest.mark.parametrize(
    ("name", "has_state"),
    [
        pytest.param("small", False, id="short-defaults"),
        pytest.param("small", True, id="short-initial-state"),
        pytest.param("packed-small", False, id="packed-short"),
        pytest.param("packed-partial", True, id="packed-partial-state"),
        pytest.param("dense-H4", False, id="dense-h4"),
        pytest.param("dense-G4", True, id="dense-grouped-state"),
        pytest.param("dense-H16", False, id="dense-h16-zero"),
        pytest.param("dense-H16", True, id="dense-h16-state"),
        pytest.param("packed-disparate", True, id="packed-disparate"),
        pytest.param("packed-G2", False, id="packed-grouped"),
        pytest.param("packed-long", False, id="packed-long-zero"),
        pytest.param("packed-long", True, id="packed-long-state"),
        pytest.param("dense-long", False, id="dense-long-zero"),
        pytest.param("dense-long", True, id="dense-long-state"),
    ],
)
def test_chunk_qla_ptx(name, has_state, monkeypatch):
    pytest.importorskip("tvm_ffi")
    assert QLAPTXBackend.is_available()
    args = inputs(name, has_state)
    selected = []
    implementation = QLAPTXBackend.chunk_gated_delta_rule

    def forward(self, *args, **kwargs):
        selected.append(True)
        return implementation(self, *args, **kwargs)

    monkeypatch.setattr(QLAPTXBackend, "chunk_gated_delta_rule", forward)
    result = chunk_gated_delta_rule(**args, output_final_state=True)
    assert selected, "the public entry did not select QLA PTX"
    reference = fused_recurrent_gated_delta_rule(
        q=args["q"].float(),
        k=args["k"].float(),
        v=args["v"].float(),
        g=args["g"],
        beta=args["beta"],
        initial_state=args["initial_state"],
        cu_seqlens=args["cu_seqlens"],
        output_final_state=True,
    )
    assert_close("o", reference[0], result[0].float(), 0.005)
    assert_close("ht", reference[1], result[1], 0.005)
    if name == "small" and not has_state:
        explicit = chunk_gated_delta_rule(**args, use_gate_in_kernel=False, A_log=None, dt_bias=None, chunk_size=64)
        assert explicit[1] is None
        assert torch.equal(explicit[0], result[0])
        assert len(selected) == 2


@pytest.mark.parametrize(
    "case",
    ["gradients", "normalization", "state-layout", "dtype", "unexported-shape"],
)
def test_chunk_qla_ptx_fallback(case, monkeypatch):
    args = inputs("small", False)
    if case == "gradients":
        args["q"].requires_grad_(True)
    elif case == "normalization":
        args["use_qk_l2norm_in_kernel"] = True
    elif case == "state-layout":
        args["state_v_first"] = True
    elif case == "dtype":
        for name in ("q", "k", "v"):
            args[name] = args[name].half()
    else:
        for name in ("q", "k", "v", "g", "beta"):
            args[name] = args[name][:, :16].contiguous()
    received = []

    class Fallback(BaseBackend):
        def chunk_gated_delta_rule(self, **kwargs):
            received.append(kwargs)
            return kwargs["q"] * 2, None

    registry = _function_registries[inspect.unwrap(chunk_gated_delta_rule)]
    monkeypatch.setattr(registry, "_backends", {"qla_ptx": QLAPTXBackend(), "fallback": Fallback()})
    result, _ = chunk_gated_delta_rule(**args)
    assert received == [args]
    assert torch.equal(result, args["q"] * 2)
    if case == "gradients":
        result.sum().backward()
        assert torch.equal(args["q"].grad, torch.full_like(args["q"], 2))


def test_qla_ptx_retry(monkeypatch):
    selected = []

    def module(record):
        selected.append(record["kernel_name"])
        if len(selected) == 1:
            raise RuntimeError("transient load failure")
        return record

    monkeypatch.setattr(_runtime, "Module", module)
    chunk_async.modules.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="transient load failure"):
            chunk_async.modules(17, 2, 4, True, False, 8)
        chunk_async.modules(17, 2, 4, True, False, 8)
        assert selected == ["kernel_prepare_bf16", "kernel_prepare_bf16", "kernel_apply_async"]
    finally:
        chunk_async.modules.cache_clear()


@pytest.mark.parametrize("case", ["missing-cpu-offsets", "different-boundaries"])
def test_qla_ptx_packed_metadata(case):
    args = inputs("packed-small", False)
    if case == "missing-cpu-offsets":
        args["cu_seqlens_cpu"] = None
    else:
        args["cu_seqlens_cpu"][1] += 1
    accepted, reason = QLAPTXBackend().chunk_gated_delta_rule_verifier(**args)
    assert not accepted
    assert reason is not None
