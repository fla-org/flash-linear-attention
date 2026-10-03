# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Check binary-level geometry and measure complete log-linear attention calls."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path


def check_geometry():
    checked = 0
    for size in (1, 2, 4, 8, 16, 32, 64, 128):
        for row in range(size):
            for col in range(size):
                xor = row ^ col
                level = sum(xor >= (1 << bit) for bit in range(size.bit_length() - 1)) if row >= col else 0
                for index in range(size.bit_length()):
                    if index == 0:
                        expected = row == col
                    else:
                        width = 1 << index
                        start = row // width * width
                        midpoint = start + width // 2
                        expected = midpoint <= row < start + width and start <= col < midpoint
                    upper = 1 << index
                    actual = row >= col and level == index
                    interval = row >= col and upper // 2 <= xor < upper
                    if expected != actual or expected != interval:
                        raise RuntimeError(f"Level mismatch: {size=}, {row=}, {col=}, {index=}")
                    checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--layout', choices=('dense', 'varlen', 'packed'), default='dense')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size for the dense layout')
    parser.add_argument('--length', type=int, default=512, help='sequence length for the dense layout')
    parser.add_argument('--sequences', type=int, choices=(32, 128), default=32)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--trace', type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error('rounds and repeats must be positive')
    if args.batch_size < 1 or args.length < 1:
        parser.error('batch size and length must be positive')
    if args.geometry_only:
        print(json.dumps({'geometry_cells': check_geometry()}))
        return

    import torch
    import triton

    from fla.ops.log_linear_attn import chunk_log_linear_attn
    from fla.ops.log_linear_attn.naive import naive_log_linear_attn

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    lengths = [args.length] * args.batch_size if args.layout == 'dense' else [127, 257]
    if args.layout == 'packed':
        lengths = [31, 63, 65, 127] * (args.sequences // 4)
    batch, length = (args.batch_size, args.length) if args.layout == 'dense' else (1, sum(lengths))
    groups, heads, key_dim, value_dim = 2, 4, 128, 64
    levels = max(7, math.ceil(math.log2(max(lengths))) + 1)
    cu_seqlens = (
        torch.tensor([0, *lengths], dtype=torch.int64, device='cuda').cumsum(0)
        if args.layout != 'dense' else None
    )
    q = torch.randn(batch, length, groups, key_dim, device='cuda')
    k = torch.randn_like(q)
    dt = torch.nn.functional.softplus(torch.randn(batch, length, heads, device='cuda') - 4)
    v = torch.randn(batch, length, heads, value_dim, device='cuda') * dt[..., None]
    g = -torch.exp(torch.rand(heads, device='cuda')) * dt
    weights = torch.randn(batch, length, heads, levels, device='cuda')
    inputs = tuple(x.requires_grad_() for x in (q, k, v, g, weights))
    do = torch.randn_like(v)

    def forward():
        return chunk_log_linear_attn(*inputs, cu_seqlens=cu_seqlens, scale=1.0)[0]

    def forward_backward():
        return torch.autograd.grad(forward(), inputs, do)

    reference_inputs = tuple(x.detach().clone().requires_grad_() for x in inputs)
    if cu_seqlens is None:
        reference = naive_log_linear_attn(*reference_inputs, scale=1.0)
    else:
        outputs = []
        start = 0
        for count in lengths:
            outputs.append(naive_log_linear_attn(*(x[:, start:start + count] for x in reference_inputs), scale=1.0))
            start += count
        reference = torch.cat(outputs, dim=1)
    actual = forward()
    reference_gradients = torch.autograd.grad(reference, reference_inputs, do)
    actual_gradients = torch.autograd.grad(actual, inputs, do)
    errors = {}
    # these are the existing test_log_linear_attn.py forward/backward RMS gates.
    for name, expected, observed, tolerance in zip(
        ('o', 'dq', 'dk', 'dv', 'dg', 'dl'),
        (reference, *reference_gradients),
        (actual, *actual_gradients),
        (0.004, 0.007, 0.008, 0.007, 0.015, 0.015),
    ):
        expected, observed = expected.detach().double(), observed.detach().double()
        denominator = expected.square().mean().sqrt().item()
        difference = (expected - observed).square().mean().sqrt().item()
        error = difference / denominator if denominator else difference
        if not math.isfinite(error) or error > tolerance:
            raise RuntimeError(f'{name}: RMS error {error} exceeds existing gate {tolerance}')
        errors[name] = error
    del reference, actual, reference_gradients, actual_gradients, reference_inputs

    measurements = {}
    for mode, run in (('fwd', forward), ('fwd_bwd', forward_backward)):
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.rounds):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start_event.record()
            for _ in range(args.repeats):
                run()
            end_event.record()
            end_event.synchronize()
            wall_us = (time.perf_counter() - wall_start) * 1e6 / args.repeats
            event_us = start_event.elapsed_time(end_event) * 1e3 / args.repeats
            samples.append({'wall_us': wall_us, 'event_us': event_us})
        measurements[mode] = {
            'samples': samples,
            'wall_median_us': statistics.median(x['wall_us'] for x in samples),
            'event_median_us': statistics.median(x['event_us'] for x in samples),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        }

    kernels = {}
    if args.trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(args.repeats):
                forward_backward()
            torch.cuda.synchronize()
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))
        for event in profile.events():
            if event.device_type.name == 'CUDA':
                kernels.setdefault(event.name, []).append(event.device_time_total)

    result = {
        'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'triton': triton.__version__,
        'cuda': torch.version.cuda,
        'layout': args.layout,
        'lengths': lengths,
        'shape': {'G': groups, 'H': heads, 'K': key_dim, 'V': value_dim, 'L': levels},
        'dtype': str(q.dtype),
        'reference_rms_errors': errors,
        'repeats': args.repeats,
        'measurements': measurements,
        'profile_device_event_us': kernels,
        'timing_scope': 'eager complete call including host preparation; CUDA events include host-launch gaps',
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + '\n')


if __name__ == '__main__':
    main()
