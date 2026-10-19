# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared context-parallel operator measurements for GDN and KDA."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.utils.distributed import (
    add_cp_arguments,
    all_gather,
    generate_cu_seqlens,
    measure_latency,
    print_tensor_error,
    profile_function,
    qkvo_all_to_all,
    run_distributed,
)
from benchmarks.utils.reporting import BenchmarkResult

DTYPE = torch.bfloat16
B, H, K, V = 1, 64, 128, 128


def _take_gradients(*tensors):
    grads = []
    for t in tensors:
        grads.append(t.grad)
        t.grad = None
    if len(grads) == 1:
        grads = grads[0]
    return grads


def benchmark_ops(args):
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.kda import chunk_kda

    results = []
    device = torch.cuda.current_device()
    group = args.group
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    cu_seqlens = generate_cu_seqlens(args.seqlen, args.mean, args.std)
    T = args.seqlen // world_size

    q = torch.randn(B, T, H, K, dtype=DTYPE, device=device).requires_grad_(True)
    k = F.normalize(torch.randn(B, T, H, K, dtype=DTYPE, device=device), p=2, dim=-1).requires_grad_(True)
    v = torch.randn(B, T, H, V, dtype=DTYPE, device=device).requires_grad_(True)
    beta = torch.rand(B, T, H, dtype=DTYPE, device=device).sigmoid().requires_grad_(True)
    if not args.kda:
        # if / 10000., the max diff is not 0
        g = (F.logsigmoid(torch.rand(B, T, H, dtype=DTYPE, device=device)) / 1.).requires_grad_(True)
    else:
        g = (F.logsigmoid(torch.rand(B, T, H, K, dtype=DTYPE, device=device)) / 1.).requires_grad_(True)
    do = torch.randn(B, T, H, V, dtype=DTYPE, device=device)

    total_q = all_gather(q.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_k = all_gather(k.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_v = all_gather(v.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_beta = all_gather(beta.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_g = all_gather(g.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_do = all_gather(do.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)

    backward = args.backward
    op = chunk_gated_delta_rule if not args.kda else chunk_kda

    def single_device():
        total_out, total_ht = op(
            total_q, total_k, total_v, total_g, total_beta,
            cu_seqlens=cu_seqlens
        )
        dist.barrier()
        if backward:
            total_out.backward(total_do)
        dist.barrier()
        return total_out.chunk(world_size, 1)[rank]

    def context_parallel():
        cp_context = build_cp_context(cu_seqlens, group)
        o, ht = op(
            q, k, v, g, beta,
            cu_seqlens=cp_context.cu_seqlens,
            cp_context=cp_context,
        )
        dist.barrier()
        if backward:
            o.backward(do)
            dist.barrier()
        return o

    def all_to_all():
        q2, k2, v2 = [qkvo_all_to_all(t.squeeze(0), is_qkv=True, group=group)[0].unsqueeze(0) for t in [q, k, v]]
        if not args.kda:
            g2 = qkvo_all_to_all(g.squeeze(0).unsqueeze(-1), is_qkv=True, group=group)[0].unsqueeze(0).squeeze(-1)
        else:
            g2 = qkvo_all_to_all(g.squeeze(0), is_qkv=True, group=group)[0].unsqueeze(0)
        beta2 = qkvo_all_to_all(beta.squeeze(0).unsqueeze(-1), is_qkv=True, group=group)[0].unsqueeze(0).squeeze(-1)
        o, ht = op(
            q2, k2, v2, g2, beta2,
            cu_seqlens=cu_seqlens
        )
        o = qkvo_all_to_all(o.squeeze(0), is_qkv=False, group=group)[0].unsqueeze(0)
        dist.barrier()
        if backward:
            o.backward(do)
            dist.barrier()
        return o

    o = context_parallel()
    dq, dk, dv, dg, dbeta = _take_gradients(q, k, v, g, beta)
    ref_o = all_to_all()
    ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta = _take_gradients(q, k, v, g, beta)

    if rank == 0:
        print(cu_seqlens)
    dist.barrier()
    print_tensor_error(o, ref_o, f"rank:{rank}, out:")
    dist.barrier()
    if backward:
        print_tensor_error(dq, ref_dq, f"rank:{rank}, dq:")
        dist.barrier()
        print_tensor_error(dk, ref_dk, f"rank:{rank}, dk:")
        dist.barrier()
        print_tensor_error(dv, ref_dv, f"rank:{rank}, dv:")
        dist.barrier()
        print_tensor_error(dg, ref_dg, f"rank:{rank}, dg:")
        dist.barrier()
        print_tensor_error(dbeta, ref_dbeta, f"rank:{rank}, dbeta:")
        dist.barrier()

    dist.barrier()
    t1 = measure_latency(single_device)
    dist.barrier()
    t2 = measure_latency(context_parallel)
    dist.barrier()
    t3 = measure_latency(all_to_all)
    for provider, elapsed in (('non_cp', t1), ('custom_cp', t2), ('all_to_all', t3)):
        results.append(BenchmarkResult(
            name='kda' if args.kda else 'gdn',
            parameters={'scope': 'ops', 'provider': provider, 'B': B, 'T': args.seqlen, 'H': H, 'K': K, 'V': V,
                        'world_size': world_size, 'backward': backward},
            metrics={'mean_ms': elapsed},
            primary_metric='mean_ms',
        ))

    if args.profile:
        def fn():
            single_device()
            dist.barrier()
            context_parallel()
            dist.barrier()
            all_to_all()
            dist.barrier()
        profile_function(fn, args.profile_path)

    return results


def add_arguments(parser):
    add_cp_arguments(parser)
    parser.add_argument('--op', choices=['chunk_gdn', 'chunk_kda'], default='chunk_gdn')


def run(args):
    args.kda = args.op == 'chunk_kda'
    return run_distributed(args=args, measure=benchmark_ops)
