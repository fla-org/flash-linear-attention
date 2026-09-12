# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend NPUGraph capture/replay coverage for KDA training."""

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda
from fla.utils import IS_NPU, device

pytestmark = pytest.mark.skipif(
    not IS_NPU or not hasattr(torch, "npu") or not hasattr(torch.npu, "make_graphed_callables"),
    reason="KDA NPUGraph tests require an Ascend NPU with make_graphed_callables",
)

T = 128
H = 2
D = 64
N = 2
DTYPE = torch.float16
_INPUT_NAMES = ("q", "k", "v", "g", "beta", "h0")
_OUTPUT_NAMES = ("o", "ht", "dq", "dk", "dv", "dg", "db", "dh0")


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
