# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse
import os
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist

from fla.modules.conv.causal_conv1d import causal_conv1d
from fla.ops.cp import build_cp_context
from fla.utils import get_abs_err, get_err_ratio

conv = torch.compiler.disable(causal_conv1d)


def run_conv(x, w, offsets, backend, cp_context=None):
    return conv(
        x=x.unsqueeze(0),
        weight=w,
        activation='silu',
        cu_seqlens=offsets.offsets,
        cu_seqlens_cpu=offsets.offsets_cpu,
        backend=backend,
        cp_context=cp_context,
    )[0]


def make_offsets(lengths):
    cpu = torch.nn.functional.pad(torch.tensor(lengths).cumsum(0, dtype=torch.int32), (1, 0))
    return SimpleNamespace(offsets=cpu.cuda(), offsets_cpu=cpu)


def compare_backends(x, w, dy, offsets, context, cp):
    outputs = []
    for backend in ('triton', 'cuda'):
        y = run_conv(x, w, offsets, backend, context)
        dx, dw = torch.autograd.grad(y, (x, w), dy)
        if cp:
            dist.all_reduce(dw)
        outputs.append((y.detach(), dx, dw))
    for name, baseline, candidate in zip(('output', 'dx', 'dw'), *outputs):
        baseline, candidate = baseline.float(), candidate.float()
        errors = torch.tensor(
            [get_abs_err(baseline, candidate), get_err_ratio(baseline, candidate)], device=baseline.device,
        )
        if cp:
            dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if not cp or dist.get_rank() == 0:
            print(f"  {name:6s} CUDA vs Triton: max_abs={errors[0].item():.3e}, "
                  f"relative_rms={errors[1].item():.3e}", flush=True)


def benchmark_lengths(tokens, packed, cp):
    lengths = [tokens] if not packed else [1, 2, 61, tokens // 4 - 64, tokens // 4, tokens // 2]
    if cp and packed:
        lengths[-2] += 7
        lengths[-1] -= 7
    return lengths


def bench_case(tokens, dim, packed, repeats, iterations, cp=False):
    torch.manual_seed(42)
    global_tokens = tokens * dist.get_world_size() if cp else tokens
    lengths = benchmark_lengths(global_tokens, packed, cp)
    x = torch.randn(tokens, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(dim, 4, device='cuda', dtype=torch.bfloat16) / 2).requires_grad_()
    dy = torch.randn(1, tokens, dim, device='cuda', dtype=x.dtype)
    offsets = make_offsets(lengths)
    context = build_cp_context(offsets.offsets, group=dist.group.WORLD, conv1d_kernel_size=4,
                               cu_seqlens_cpu=offsets.offsets_cpu) if cp else None
    if not cp or dist.get_rank() == 0:
        print(f"\nTokens/rank={tokens}, channels={dim}, packed={packed}", flush=True)
    compare_backends(x, w, dy, offsets, context, cp)
    for backward in (False, True):
        def step(backend, backward=backward):
            y = run_conv(x, w, offsets, backend, context)
            return torch.autograd.grad(y, (x, w), dy) if backward else y

        # warmup
        for backend in ('triton', 'cuda'):
            for _ in range(5):
                step(backend)
        torch.cuda.synchronize()
        samples = {'triton': [], 'cuda': []}
        for _ in range(repeats):
            for backend in ('triton', 'cuda'):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                if cp:
                    dist.barrier()
                start.record()
                for _ in range(iterations):
                    step(backend)
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end) / iterations
                if cp:
                    value = torch.tensor(elapsed, device='cuda')
                    dist.all_reduce(value, op=dist.ReduceOp.MAX)
                    elapsed = value.item()
                samples[backend].append(elapsed)
        if not cp or dist.get_rank() == 0:
            mode = 'forward_backward' if backward else 'forward'
            for label, aggregate in (('avg', statistics.mean), ('median', statistics.median)):
                triton_ms, cuda_ms = (aggregate(samples[b]) for b in ('triton', 'cuda'))
                print(f"  {mode:17s} {label:6s}  Triton {triton_ms:.3f} ms  CUDA {cuda_ms:.3f} ms  "
                      f"speedup {triton_ms / cuda_ms:.3f}x", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cp', action='store_true')
    p.add_argument('--tokens', type=int, nargs='+', default=[2048, 8192, 16384, 32768])
    p.add_argument('--channels', type=int, nargs='+', default=[2048, 4096, 8192])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--iterations', type=int, default=10)
    args = p.parse_args()
    if min(args.tokens) < 256 or min(args.channels + [args.repeats, args.iterations]) < 1:
        p.error('tokens must be >= 256; channels, repeats and iterations must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if args.cp:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl')
        if dist.get_world_size() < 2:
            p.error('CP benchmark requires at least two ranks')
    if not args.cp or dist.get_rank() == 0:
        print(f"GPU: {torch.cuda.get_device_name()}, CP: {dist.get_world_size() if args.cp else 1}", flush=True)

    for tokens in args.tokens:
        for dim in args.channels:
            for packed in (False, True):
                bench_case(tokens, dim, packed, args.repeats, args.iterations, args.cp)
    if args.cp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
