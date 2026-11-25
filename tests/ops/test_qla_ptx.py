# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

"""Validate the fixed PTX assets, retry binding, and explicit forward runner."""

import hashlib
import json
import math

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.backends.qla_ptx import _runtime, gdn_cp, run
from fla.utils import assert_close, device, get_device_capability


def test_qla_ptx_assets():
    manifest = json.loads((_runtime.ROOT / "manifest.json").read_text())
    assert manifest["complete"] and manifest["arch"] == "sm_103a"
    assert len(manifest["bindings"]) == 123
    for record in manifest["bindings"].values():
        for key in ("ptx", "shim"):
            assert hashlib.sha256((_runtime.ROOT / record[key]).read_bytes()).hexdigest() == record[key + "_sha256"]


def test_qla_ptx_retry(monkeypatch):
    selected = []

    def module(record):
        selected.append(record["kernel_name"])
        if len(selected) == 2:
            raise RuntimeError("transient load failure")
        return record["kernel_name"]

    monkeypatch.setattr(_runtime, "Module", module)
    gdn_cp.modules.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="transient load failure"):
            gdn_cp.modules(2, 8, 8, 1)
        result = gdn_cp.modules(2, 8, 8, 1)
        assert result == ("kernel_cp_state", "kernel_cp_comb", "kernel_cp_out")
        assert selected == ["kernel_cp_state", "kernel_cp_comb", *result]
    finally:
        gdn_cp.modules.cache_clear()


@pytest.mark.skipif(device != "cuda" or get_device_capability() != (10, 3), reason="requires NVIDIA SM103")
@pytest.mark.parametrize(
    "name", ["m00_s04", "m00_s00", "m02_s10", "m03_s03", "m04_s05", "m05_s15", "m06_s06", "m07_s09", "m08_s12"]
)
@torch.inference_mode()
def test_qla_ptx_forward(name):
    pytest.importorskip("tvm_ffi")
    torch.manual_seed(42)
    config = json.loads((_runtime.ROOT / "benchmark_shared.json").read_text())
    w = next(w for w in config["workloads"] if w["name"] == name)
    t, hq, hv = (w[key] for key in ("T", "Hq", "Hv"))
    q, k = [F.normalize(torch.randn(1, t, hq, 128, device=device), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(1, t, hv, 128, device=device, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, t, hv, device=device)) / 16
    mask = torch.zeros(hv, dtype=torch.bool, device=device)
    mask[: math.ceil(0.75 * hv)] = True
    mask = mask[torch.randperm(hv, device=device)]
    g[:, :, ~mask] = 0
    beta = torch.randn(1, t, hv, device=device).sigmoid()
    initial_state = torch.randn(len(w["lengths"]), hv, 128, 128, device=device)
    cu_seqlens = torch.tensor(w["cu"], dtype=torch.int32, device=device)
    actual = run(q, k, v, g, beta, 128**-0.5, initial_state, cu_seqlens)
    expected = fused_recurrent_gated_delta_rule(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    assert actual[0].dtype == torch.bfloat16 and actual[1].dtype == torch.float32
    assert torch.isfinite(actual[0]).all() and torch.isfinite(actual[1]).all()
    assert_close("output", expected[0], actual[0].float(), 0.005)
    assert_close("final_state", expected[1], actual[1], 0.005)
