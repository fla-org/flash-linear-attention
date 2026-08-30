# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Graph capture/replay tests for KDA chunk training (fwd + bwd).

Captures a fwd+bwd step once, replays it with different ``cu_seqlens`` contents
(different splits and zero-length tail padding), and compares against the eager
(``use_graph=False``) path. Requires CUDA + CUDAGraph.
"""

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda
from fla.utils import assert_close, device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch.cuda, "CUDAGraph"),
    reason="graph capture tests require a CUDA device with CUDAGraph support",
)

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
