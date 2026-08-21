# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse
import statistics

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark reusable final_state for fused recurrent GDN.')
    parser.add_argument('--B', type=int, default=1)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--H', type=int, default=32)
    parser.add_argument('--HV', type=int, default=32)
    parser.add_argument('--K', type=int, default=128)
    parser.add_argument('--V', type=int, default=128)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iters', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark.')
    if args.HV % args.H != 0:
        raise ValueError(f'HV ({args.HV}) must be divisible by H ({args.H}).')

    device = torch.device('cuda')
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(42)
    q = torch.randn(args.B, args.T, args.H, args.K, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn(args.B, args.T, args.HV, args.V, dtype=dtype, device=device)
    beta = torch.sigmoid(torch.randn(args.B, args.T, args.HV, dtype=dtype, device=device))
    g = F.logsigmoid(torch.randn(args.B, args.T, args.HV, dtype=torch.float32, device=device))
    initial_state = torch.randn(args.B, args.HV, args.K, args.V, dtype=torch.float32, device=device)
    final_state = torch.empty_like(initial_state)

    def run(use_buffer):
        kwargs = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        if use_buffer:
            kwargs['final_state'] = final_state
        return fused_recurrent_gated_delta_rule(**kwargs)

    ref_o, ref_ht = run(False)
    buf_o, buf_ht = run(True)
    torch.testing.assert_close(ref_o, buf_o, rtol=0.002, atol=0.002)
    torch.testing.assert_close(ref_ht, buf_ht, rtol=0.002, atol=0.002)
    if buf_ht.data_ptr() != final_state.data_ptr():
        raise AssertionError('final_state buffer was not reused.')

    def measure(use_buffer):
        for _ in range(args.warmup):
            run(use_buffer)
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                run(use_buffer)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / args.iters)
        return samples

    baseline = measure(False)
    buffered = measure(True)
    baseline_median = statistics.median(baseline)
    buffered_median = statistics.median(buffered)
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'shape: B={args.B}, T={args.T}, H={args.H}, HV={args.HV}, K={args.K}, V={args.V}, dtype={args.dtype}')
    print(f'baseline ms/call: {baseline}')
    print(f'buffered ms/call: {buffered}')
    print(f'baseline median: {baseline_median:.6f} ms')
    print(f'buffered median: {buffered_median:.6f} ms')
    print(f'speedup: {baseline_median / buffered_median:.4f}x')
    print(f'final_state data_ptr: {final_state.data_ptr()}')


if __name__ == '__main__':
    main()
