# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
KDA context-parallel performance on Ascend NPU.

Same operator comparison as ``benchmarks/cp/test_gdn_with_cp.py --ops --kda``:
every rank holds a local slice, then three timed paths run.

* single: inputs are all-gathered before the timer. Each rank runs ``chunk_kda``
  on the full sequence and all heads.
* custom CP: local sequence, all heads, ``build_cp_context``.
* all-to-all: exchange sequence for heads, run the full sequence on ``H / world``
  heads, then exchange back.

``torch.npu.Event`` times the calls. Collectives inside the timed function make
``triton.testing.do_bench`` report the wrong interval. ``profile_npu.py`` is
single-process and deadlocks here; rank 0 traces while every rank issues the
same number of calls.

Usage:
    torchrun --nproc_per_node=4 benchmarks/cp/test_kda_with_cp.py \\
        --seqlen 32768 --backward

    torchrun --nproc_per_node=2 benchmarks/cp/test_kda_with_cp.py \\
        --seqlen 8192 --forward-only --profile --no-bench
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise SystemExit("torch_npu is required to benchmark KDA CP on NPU.") from exc

from torch_npu.profiler import (
    AiCMetrics,
    ExportType,
    ProfilerActivity,
    ProfilerLevel,
    _ExperimentalConfig,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda

DTYPE = torch.bfloat16
B = 1
H = 64
K = 128
V = 128

# stopping the Ascend profiler on RECORD_AND_SAVE skips operator_details.csv.
# repeat=1 makes the step after the active window NONE.
PROFILE_WARMUP = 2
PROF_WAIT = 0
PROF_SCHEDULE_WARMUP = 1
PROF_ACTIVE = 5
PROF_REPEAT = 1


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark chunk_kda CP on Ascend NPU")
    parser.add_argument("--seqlen", type=int, default=32768, help="Total sequence length")
    parser.add_argument("--mean", type=int, default=32768, help="Mean sequence length for cu_seqlens")
    parser.add_argument("--std", type=int, default=0, help="Sequence-length spread; 0 is one sequence")
    parser.add_argument("--backward", action="store_true", help="Time forward and backward")
    parser.add_argument("--forward-only", action="store_true", help="Time forward only")
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True,
                        help="Time the three paths (default: true)")
    parser.add_argument("--profile", action="store_true", help="Trace the three paths on rank 0")
    parser.add_argument("--profile-dir", type=str, default="profile/kda_cp_npu",
                        help="Directory for the rank-0 NPU trace")
    parser.add_argument("--steps", type=int, default=20, help="Timed steps")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warmup steps")
    return parser.parse_args()


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            print(*args, **kwargs)
        dist.barrier()
    else:
        print(*args, **kwargs)


def all_gather(x, group=None) -> torch.Tensor:
    # HCCL rejects a non-contiguous input.
    x = x.contiguous()
    world_size = dist.get_world_size(group=group)
    y = torch.empty(world_size * x.size(0), *x.shape[1:], device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(y, x, group=group)
    return y


def generate_cu_seqlens(end, mean, var, device):
    random.seed(42)
    offsets = [0]
    while offsets[-1] < end:
        length = random.randint(max(mean - var, 1), mean + var)
        offsets.append(offsets[-1] + length)
    offsets[-1] = end
    return torch.tensor(offsets, device=device, dtype=torch.int32)


def a2a(x, stage=1, group=None):
    x = x.contiguous()
    world_size = dist.get_world_size(group)
    tokens, heads, dim = x.shape
    if stage == 1:
        x = x.reshape(tokens, world_size, heads // world_size, dim).transpose(0, 1).contiguous()
    else:
        x = x.reshape(world_size, tokens // world_size, heads, dim).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    if stage == 1:
        return out.flatten(0, 1)
    return out.transpose(0, 1).contiguous().flatten(1, 2)


class _All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stages, group):
        fwd_stage, bwd_stage = stages
        ctx.bwd_stage = bwd_stage
        ctx.group = group
        return a2a(x, fwd_stage, group)

    @staticmethod
    def backward(ctx, grad):
        return a2a(grad, stage=ctx.bwd_stage, group=ctx.group), None, None


def qkvo_all2all(x, is_qkv=True, group=None):
    stages = (1, 2) if is_qkv else (2, 1)
    return _All2All.apply(x, stages, group)


def bench(fn, step=20, warm_up=10):
    """Time one step with NPU events. The interval stays open across ``step`` calls."""
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    for _ in range(warm_up):
        fn()
    torch.npu.synchronize()
    start_event.record()
    for _ in range(step):
        fn()
    end_event.record()
    torch.npu.synchronize()
    return start_event.elapsed_time(end_event) / step


def compare(x, y, prefix=""):
    if x is None or y is None:
        return
    diff = (x.float() - y.float()).abs()
    dist.barrier()
    for rank in range(dist.get_world_size()):
        if rank == dist.get_rank():
            absmax = torch.maximum(x.abs().max(), y.abs().max()).item()
            print(f"{prefix}max_diff: {diff.max().item()}, mean_diff: {diff.mean().item()}, absmax: {absmax}")
        dist.barrier()


def clear_grad(*tensors):
    grads = []
    for tensor in tensors:
        grads.append(None if tensor.grad is None else tensor.grad.detach())
        tensor.grad = None
    return grads[0] if len(grads) == 1 else grads


def _run_calls(fn, count):
    for _ in range(count):
        fn()
    torch.npu.synchronize()


def _profile_fn_calls(steps=PROF_ACTIVE, warmup=PROFILE_WARMUP):
    """Calls inside ``profile_kernels``. The closing ``prof.step()`` does not call ``fn``."""
    return warmup + steps + 1


def profile_kernels(fn, out_dir, steps=PROF_ACTIVE, warmup=PROFILE_WARMUP):
    """Trace ``steps`` iterations on this rank and return operator rows."""
    _run_calls(fn, warmup)
    experimental_config = _ExperimentalConfig(
        profiler_level=ProfilerLevel.Level1,
        aic_metrics=AiCMetrics.PipeUtilization,
        export_type=ExportType.Text,
    )
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=schedule(
            wait=PROF_WAIT,
            warmup=PROF_SCHEDULE_WARMUP,
            active=steps,
            repeat=PROF_REPEAT,
        ),
        on_trace_ready=tensorboard_trace_handler(out_dir),
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(steps + 1):
            torch.npu.synchronize()
            fn()
            torch.npu.synchronize()
            prof.step()
        prof.step()
    return _load_operator_stats(out_dir)


def _load_operator_stats(root):
    paths = sorted(Path(root).rglob("operator_details.csv"))
    if not paths:
        return []
    aggregated = {}
    with paths[-1].open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                device_us = float(row.get("Device Self Duration(us)") or 0)
            except ValueError:
                continue
            if device_us <= 0:
                continue
            name = row.get("Name") or ""
            stat = aggregated.setdefault(name, {"name": name, "device_time_ms": 0.0, "calls": 0})
            stat["device_time_ms"] += device_us / 1000
            stat["calls"] += 1
    stats = list(aggregated.values())
    stats.sort(key=lambda item: item["device_time_ms"], reverse=True)
    return stats


def format_kernel_table(kernel_stats, top_n=20):
    if not kernel_stats:
        return "No NPU operator rows with device time. See the trace directory."
    total_time = sum(item["device_time_ms"] for item in kernel_stats)
    lines = [
        f"{'Rank':<4} {'Operator':<60} {'Time (ms)':<12} {'Calls':<8} {'%':<6}",
        "-" * 96,
    ]
    for index, stat in enumerate(kernel_stats[:top_n]):
        pct = stat["device_time_ms"] / total_time * 100 if total_time > 0 else 0
        lines.append(
            f"{index + 1:<4} {stat['name'][:58]:<60} {stat['device_time_ms']:<12.3f} "
            f"{stat['calls']:<8} {pct:<6.1f}"
        )
    lines.append("-" * 96)
    lines.append(f"{'Total':<65} {total_time:<12.3f}")
    return "\n".join(lines)


def test_ops(args):
    device = torch.npu.current_device()
    group = args.group
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if H % world_size != 0:
        raise ValueError(f"H={H} must be divisible by world_size={world_size} for all-to-all")
    if args.seqlen % world_size != 0:
        raise ValueError(f"seqlen={args.seqlen} must be divisible by world_size={world_size}")

    cu_seqlens = generate_cu_seqlens(args.seqlen, args.mean, args.std, device)
    local_t = args.seqlen // world_size
    q = torch.randn(B, local_t, H, K, dtype=DTYPE, device=device).requires_grad_(True)
    k = F.normalize(torch.randn(B, local_t, H, K, dtype=DTYPE, device=device), p=2, dim=-1).requires_grad_(True)
    v = torch.randn(B, local_t, H, V, dtype=DTYPE, device=device).requires_grad_(True)
    beta = torch.rand(B, local_t, H, dtype=DTYPE, device=device).sigmoid().requires_grad_(True)
    # pre-gated log-space decay, same call as test_gdn_with_cp.py --ops --kda.
    g = (F.logsigmoid(torch.rand(B, local_t, H, K, dtype=DTYPE, device=device))).requires_grad_(True)
    do = torch.randn(B, local_t, H, V, dtype=DTYPE, device=device)

    total_q = all_gather(q.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_k = all_gather(k.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_v = all_gather(v.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_beta = all_gather(beta.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_g = all_gather(g.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_do = all_gather(do.squeeze(0), group=group).unsqueeze(0).detach()
    backward = args.backward and not args.forward_only

    def kda_no_cp():
        total_out, _ = chunk_kda(total_q, total_k, total_v, total_g, total_beta, cu_seqlens=cu_seqlens)
        dist.barrier()
        if backward:
            total_out.backward(total_do)
        dist.barrier()
        return total_out.chunk(world_size, 1)[rank]

    def kda_with_custom_cp():
        cp_context = build_cp_context(cu_seqlens, group)
        out, _ = chunk_kda(
            q, k, v, g, beta,
            cu_seqlens=cp_context.cu_seqlens,
            cp_context=cp_context,
        )
        dist.barrier()
        if backward:
            out.backward(do)
            dist.barrier()
        return out

    def kda_with_a2a():
        q2, k2, v2, g2 = [qkvo_all2all(t.squeeze(0), is_qkv=True, group=group).unsqueeze(0) for t in (q, k, v, g)]
        beta2 = qkvo_all2all(beta.squeeze(0).unsqueeze(-1), is_qkv=True, group=group).unsqueeze(0).squeeze(-1)
        out, _ = chunk_kda(q2, k2, v2, g2, beta2, cu_seqlens=cu_seqlens)
        out = qkvo_all2all(out.squeeze(0), is_qkv=False, group=group).unsqueeze(0)
        dist.barrier()
        if backward:
            out.backward(do)
            dist.barrier()
        return out

    out = kda_with_custom_cp()
    dq, dk, dv, dg, dbeta = clear_grad(q, k, v, g, beta)
    ref = kda_with_a2a()
    ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta = clear_grad(q, k, v, g, beta)
    print_rank0(cu_seqlens)
    compare(out, ref, f"rank:{rank}, out:")
    if backward:
        compare(dq, ref_dq, f"rank:{rank}, dq:")
        compare(dk, ref_dk, f"rank:{rank}, dk:")
        compare(dv, ref_dv, f"rank:{rank}, dv:")
        compare(dg, ref_dg, f"rank:{rank}, dg:")
        compare(dbeta, ref_dbeta, f"rank:{rank}, dbeta:")

    if args.bench:
        dist.barrier()
        single_ms = bench(kda_no_cp, step=args.steps, warm_up=args.warmup)
        dist.barrier()
        custom_ms = bench(kda_with_custom_cp, step=args.steps, warm_up=args.warmup)
        dist.barrier()
        a2a_ms = bench(kda_with_a2a, step=args.steps, warm_up=args.warmup)
        dist.barrier()
        if rank == world_size - 1:
            print(
                f"custom cp, non-cp time: {single_ms:.3f} ms, cp time: {custom_ms:.3f} ms, "
                f"rate: {((single_ms / world_size) / custom_ms * 100):.2f} %"
            )
            print(
                f"all to all, non-cp time: {single_ms:.3f} ms, cp time: {a2a_ms:.3f} ms, "
                f"rate: {((single_ms / world_size) / a2a_ms * 100):.2f} %"
            )

    if args.profile:
        def traced():
            kda_no_cp()
            dist.barrier()
            kda_with_custom_cp()
            dist.barrier()
            kda_with_a2a()
            dist.barrier()

        if rank == 0:
            stats = profile_kernels(traced, args.profile_dir)
            print(format_kernel_table(stats))
            print(f"trace directory: {args.profile_dir}")
        else:
            _run_calls(traced, _profile_fn_calls())


def main():
    args = get_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # set the device before HCCL init. Passing device_id into init_process_group
    # fails to bind the NPU socket port.
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    torch.manual_seed(rank + 42)
    torch.npu.manual_seed(rank + 42)
    random.seed(42)
    # HCCL builds a communicator on the first collective of a new group.
    # Another group of the same ranks times out waiting for that handshake.
    args.group = dist.group.WORLD
    test_ops(args)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
