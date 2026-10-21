# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Test for Context Parallel (CP) KDA with platform graph capture/replay.

Each rank captures a fwd+bwd step (cp_context + use_graph) once, replays it with
several cu_seqlens layouts (different splits and zero-length tail padding), and
compares against the eager CP path on the same shard. The recorded region includes
the cross-rank state all-gathers, so all ranks must capture and replay in lockstep.
"""


import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.ops.cp import FLACPContext, build_cp_context
from fla.ops.kda import chunk_kda
from fla.utils import IS_AMD, IS_NVIDIA, assert_close, device
from tests.ops.kda_graph import MAX_NUM_SEQS, VARLEN_CASES, D, H, T, make_inputs

pytestmark = pytest.mark.skipif(
    not (IS_NVIDIA or IS_AMD), reason="direct graph capture requires CUDA or HIP",
)

TOKEN_NAMES = ["q", "k", "v", "g", "beta"]
INPUT_NAMES = TOKEN_NAMES + ["A_log", "dt_bias"]
OUTPUT_NAMES = ["o", "dq", "dk", "dv", "dg", "db", "dA", "dbias"]


def _make_inputs(seed, device):
    """Same seed on every rank reproduces one logical global batch; ranks slice their shard."""
    g = torch.Generator(device).manual_seed(seed)
    dt = torch.bfloat16
    hv = H
    q = torch.randn(1, T, H, D, dtype=dt, device=device, generator=g)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32, device=device, generator=g), p=2, dim=-1).to(dt)
    v = torch.randn(1, T, hv, D, dtype=dt, device=device, generator=g)
    gg = torch.randn(1, T, hv, D, dtype=dt, device=device, generator=g)
    A_log = torch.log(torch.empty(1, 1, hv, 1, device=device).uniform_(1, 16, generator=g))
    dt_bias = torch.randn(hv * D, dtype=torch.float32, device=device, generator=g)
    beta = torch.rand(1, T, hv, dtype=dt, device=device, generator=g).sigmoid()
    do = torch.randn(1, T, hv, D, dtype=dt, device=device, generator=g)
    return dict(q=q, k=k, v=v, g=gg, beta=beta, A_log=A_log, dt_bias=dt_bias, do=do)


def _run_graph(rank, world_size):
    device = torch.device('cuda', rank)
    part = T // world_size
    lo, hi = rank * part, (rank + 1) * part
    group = dist.group.WORLD

    cu_static = torch.zeros(MAX_NUM_SEQS + 1, dtype=torch.long, device=device)
    pre_dev = torch.zeros(1, dtype=torch.int32, device=device)
    post_dev = torch.zeros(1, dtype=torch.int32, device=device)

    def refresh(base):
        local = base.cu_seqlens
        cu_static[: len(local)] = local
        cu_static[len(local):] = local[-1]
        pre_dev.fill_(base.pre_num_ranks)
        post_dev.fill_(base.post_num_ranks)

    # one persistent context around the static buffers; host fields are only read eagerly
    base0 = build_cp_context(torch.tensor(VARLEN_CASES[0][1], dtype=torch.long, device=device), group=group)
    refresh(base0)
    ctx = FLACPContext(
        group=group,
        cu_seqlens=cu_static,
        cu_seqlens_cpu=None,
        is_last_rank=base0.is_last_rank,
        pre_num_ranks=base0.pre_num_ranks,
        is_first_rank=base0.is_first_rank,
        post_num_ranks=base0.post_num_ranks,
        pre_num_ranks_dev=pre_dev,
        post_num_ranks_dev=post_dev,
    )

    def shard_leaves(inp):
        leaves = {n: inp[n][:, lo:hi].detach().clone().requires_grad_() for n in TOKEN_NAMES}
        for n in ("A_log", "dt_bias"):
            leaves[n] = inp[n].detach().clone().requires_grad_()
        return leaves

    inp0 = _make_inputs(0, device)
    leaves = shard_leaves(inp0)
    do = inp0["do"][:, lo:hi].detach().clone()

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
            use_gate_in_kernel=True,
            A_log=leaves["A_log"],
            dt_bias=leaves["dt_bias"],
            use_graph=True,
            max_num_seqs=MAX_NUM_SEQS,
        )
        (o * do).sum().backward()
        return o

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cap_o = step()

    for seed, (tag, cucfg) in enumerate(VARLEN_CASES, start=1):
        inp = _make_inputs(seed * 17, device)
        for n in TOKEN_NAMES:
            leaves[n].data.copy_(inp[n][:, lo:hi])
        for n in ("A_log", "dt_bias"):
            leaves[n].data.copy_(inp[n])
        do.copy_(inp["do"][:, lo:hi])
        cu = torch.tensor(cucfg, dtype=torch.long, device=device)
        refresh(build_cp_context(cu, group=group))

        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        got = [cap_o.clone()] + [leaves[n].grad.clone() for n in INPUT_NAMES]
        dist.barrier()

        # eager CP reference on the same shard
        ref_leaves = shard_leaves(inp)
        o, _ = chunk_kda(
            q=ref_leaves["q"],
            k=ref_leaves["k"],
            v=ref_leaves["v"],
            g=ref_leaves["g"],
            beta=ref_leaves["beta"],
            cp_context=build_cp_context(cu, group=group),
            use_gate_in_kernel=True,
            A_log=ref_leaves["A_log"],
            dt_bias=ref_leaves["dt_bias"],
        )
        (o * inp["do"][:, lo:hi]).sum().backward()
        ref = [o.detach()] + [ref_leaves[n].grad for n in INPUT_NAMES]
        dist.barrier()

        for name, r, g in zip(OUTPUT_NAMES, ref, got):
            assert_close(f"{tag}::{name}", r, g, ratio=8e-3)

    # no destroy_process_group: tearing down a communicator recorded in a CUDA graph
    # can hang; process exit reaps it
    dist.barrier()


def test_chunk_kda(run_distributed):
    # a communicator recorded by a CUDA graph must outlive the graph; worker exit reaps it
    run_distributed(worker=_run_graph, world_size=2, destroy_process_group=False)


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


def test_chunk_kda_single_rank(single_rank_group, monkeypatch):
    # keep the eager reference on the same backend as graph capture
    monkeypatch.setenv("FLA_TILELANG", "0")
    gate = True
    cu = torch.tensor(VARLEN_CASES[0][1], dtype=torch.long, device=device)
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

    inp0 = make_inputs(seed=0, gate=gate, safe_gate=False)
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
    for seed, (tag, cucfg) in enumerate(VARLEN_CASES, start=1):
        inp = make_inputs(seed=seed * 17, gate=gate, safe_gate=False)
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
