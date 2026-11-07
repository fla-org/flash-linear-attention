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
    parser = argparse.ArgumentParser(description='Benchmark autoregressive final_state reuse for fused recurrent GDN.')
    parser.add_argument('--B', type=int, default=8)
    parser.add_argument('--H', type=int, default=32)
    parser.add_argument('--HV', type=int, default=32)
    parser.add_argument('--K', type=int, default=128)
    parser.add_argument('--V', type=int, default=256)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), default='bfloat16')
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
    q_steps = [torch.randn(args.B, 1, args.H, args.K, dtype=dtype, device=device) for _ in range(args.steps)]
    k_steps = [torch.randn_like(q) for q in q_steps]
    v_steps = [torch.randn(args.B, 1, args.HV, args.V, dtype=dtype, device=device) for _ in range(args.steps)]
    beta_steps = [torch.sigmoid(torch.randn(args.B, 1, args.HV, dtype=dtype, device=device)) for _ in range(args.steps)]
    g_steps = [
        F.logsigmoid(torch.randn(args.B, 1, args.HV, dtype=torch.float32, device=device))
        for _ in range(args.steps)
    ]
    initial_state = torch.randn(args.B, args.HV, args.K, args.V, dtype=torch.float32, device=device)

    def run_chain(use_buffer):
        state = initial_state.clone()
        for q, k, v, beta, g in zip(q_steps, k_steps, v_steps, beta_steps, g_steps):
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if use_buffer:
                kwargs['final_state'] = state
            _, state = fused_recurrent_gated_delta_rule(**kwargs)
        return state

    ref_state = run_chain(False)
    buf_state = run_chain(True)
    torch.testing.assert_close(ref_state, buf_state, rtol=0.002, atol=0.002)

    def measure_interleaved():
        samples = {False: [], True: []}
        for round_idx in range(args.rounds):
            order = (False, True) if round_idx % 2 == 0 else (True, False)
            for use_buffer in order:
                for _ in range(args.warmup):
                    run_chain(use_buffer)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_chain(use_buffer)
                end.record()
                end.synchronize()
                samples[use_buffer].append(start.elapsed_time(end))
        return samples[False], samples[True]

    baseline, buffered = measure_interleaved()
    baseline_median = statistics.median(baseline)
    buffered_median = statistics.median(buffered)
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'shape per step: B={args.B}, T=1, H={args.H}, HV={args.HV}, K={args.K}, V={args.V}, dtype={args.dtype}')
    print(f'autoregressive steps: {args.steps}')
    print(f'baseline total ms: {baseline}')
    print(f'buffered total ms: {buffered}')
    print(f'baseline median: {baseline_median:.6f} ms ({baseline_median / args.steps:.6f} ms/step)')
    print(f'buffered median: {buffered_median:.6f} ms ({buffered_median / args.steps:.6f} ms/step)')
    print(f'speedup: {baseline_median / buffered_median:.4f}x')


if __name__ == '__main__':
    main()
