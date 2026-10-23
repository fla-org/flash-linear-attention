# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare long-memory KDA CP with a tokenwise FP32 reference on NVIDIA Ampere or newer.

Run manually with `python -m benchmarks.cp.kda_precision --output /tmp/kda-precision`.
Each precision runs in a separate process with identical inputs and four intra-card subsequences.
Only the affine-chain dot precision differs between TF32 and TF32x3; ordinary KDA kernels retain their precision.
Elapsed times include compilation and are not latency benchmarks. This experiment checks forward outputs and states.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def error_metrics(reference, actual):
    import torch

    assert torch.isfinite(reference).all()
    assert torch.isfinite(actual).all()
    reference, actual = reference.float(), actual.float()
    difference = actual - reference
    rms = reference.square().mean().sqrt().item()
    assert rms > 0
    return dict(
        relative_rms=difference.square().mean().sqrt().item() / rms,
        max_absolute=difference.abs().max().item(),
        reference_rms=rms,
    )


def run_precision(mode: str, output: Path):
    import torch
    import torch.nn.functional as F
    import triton
    import triton.language as tl

    from fla.ops.common import intracard_cp
    from fla.ops.cp import chunk_bwd, chunk_fwd, chunk_merge
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.naive import naive_recurrent_kda
    from fla.utils import assert_close

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    torch.set_num_threads(1)
    lengths = [2 ** exponent for exponent in range(16, 21)]
    T, D = lengths[-1], 64
    q, k = [F.normalize(torch.randn(1, T, 1, D, device='cuda'), dim=-1).to(torch.bfloat16) for _ in range(2)]
    v = torch.randn(1, T, 1, D, device='cuda', dtype=torch.bfloat16)
    # slow decay and small updates preserve a measurable state signal over the full million-token sequence
    g = torch.full((1, T, 1, D), math.log1p(-2 ** -20), device='cuda', dtype=torch.float32)
    beta = torch.full((1, T, 1), 2 ** -16, device='cuda', dtype=torch.bfloat16)
    h0 = torch.randn(1, 1, D, D, device='cuda', dtype=torch.float32) * 0.1
    inputs = dict(q=q, k=k, v=v, g=g, beta=beta)
    rows = []
    metadata = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        seed=42,
        T=T,
        B=1,
        H=1,
        K=D,
        V=D,
        chunk_size=64,
        splits=4,
        decay=1 - 2 ** -20,
        beta=2 ** -16,
        initial_state_std=0.1,
        input_dtype='bfloat16',
        gate_dtype='float32',
        mode=mode,
    )
    print(json.dumps(metadata), flush=True)
    with torch.inference_mode():
        if mode == 'fp32':
            # the recurrent oracle uses FP32 elementwise updates and reductions, without tensor-core dots
            small = {name: value[:, :256].float().contiguous() for name, value in inputs.items()}
            ref_o, ref_h = naive_recurrent_kda(**small, initial_state=h0, output_final_state=True)
            tri_o, tri_h = fused_recurrent_kda(**small, initial_state=h0, output_final_state=True)
            oracle_check = dict(output=error_metrics(ref_o, tri_o), state=error_metrics(ref_h, tri_h))
            assert oracle_check['output']['relative_rms'] < 1e-5
            assert oracle_check['state']['relative_rms'] < 1e-5
            print(json.dumps({'oracle_check': oracle_check}), flush=True)
            metadata['oracle_check'] = oracle_check
            start = 0
            state = h0
            for length in lengths:
                segment = {name: value[:, start:length].float().contiguous() for name, value in inputs.items()}
                begin = time.monotonic()
                o, state = fused_recurrent_kda(**segment, initial_state=state, output_final_state=True)
                torch.cuda.synchronize()
                assert o.dtype == torch.float32 and state.dtype == torch.float32
                assert torch.isfinite(o).all() and torch.isfinite(state).all()
                torch.save(dict(output=o.cpu(), state=state.cpu()), output / f'reference-{length}.pt')
                row = dict(length=length, segment_start=start, seconds=time.monotonic() - begin)
                print(json.dumps(row), flush=True)
                rows.append(row)
                start = length
        else:
            for module in (chunk_fwd, chunk_bwd, chunk_merge):
                module.CP_DOT_PRECISION = tl.constexpr(mode)
            # hold CP launch configurations fixed so only affine-dot precision differs
            chunk_fwd.chunk_cp_fwd_kernel_h.fn.configs = [triton.Config({}, num_warps=4, num_stages=2)]
            chunk_merge.chunk_cp_fwd_bwd_kernel_merge.fn.configs = [triton.Config({'BV': 32}, num_warps=4, num_stages=2)]
            reference_parts = []
            for length in lengths:
                saved = torch.load(output / f'reference-{length}.pt', weights_only=True)
                reference_parts.append(saved['output'])
                ref_o = torch.cat(reference_parts, dim=1).cuda()
                ref_h = saved['state'].cuda()
                segment = {name: value[:, :length].contiguous() for name, value in inputs.items()}
                cu_cpu = torch.tensor([0, length], dtype=torch.int32)
                cu = cu_cpu.cuda()
                intracard_cp._intracard_cache.clear()
                begin = time.monotonic()
                o, ht = chunk_kda(
                    **segment,
                    initial_state=h0,
                    output_final_state=True,
                    cu_seqlens=cu,
                    cu_seqlens_cpu=cu_cpu,
                )
                torch.cuda.synchronize()
                entries = list(intracard_cp._intracard_cache.values())
                assert len(entries) == 1 and entries[0].split_info.num_subseqs == [4]
                row = dict(
                    length=length,
                    output=error_metrics(ref_o, o),
                    last_chunk=error_metrics(ref_o[:, -64:], o[:, -64:]),
                    state=error_metrics(ref_h, ht),
                    seconds=time.monotonic() - begin,
                )
                print(json.dumps(row), flush=True)
                rows.append(row)
                if mode == 'tf32x3':
                    assert_close('o', ref_o, o.float(), 0.005)
                    assert_close('ht', ref_h, ht, 0.005)
                del o, ht, ref_o, ref_h
    (output / f'{mode}.json').write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Directory for logs, metrics, and FP32 reference tensors.')
    parser.add_argument('--mode', choices=['fp32', 'tf32', 'tf32x3'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # set routing before importing FLA, including in the isolated child processes
    os.environ.update(
        FLA_CI_ENV='0',
        FLA_INTRACARD_CP='1',
        FLA_INTRACARD_MAX_SPLITS='4',
        FLA_FLASH_KDA='0',
        FLA_FLASH_QLA='0',
        FLA_TILELANG='0',
        OMP_NUM_THREADS='1',
    )
    os.environ.pop('FLA_DISABLE_BACKEND_DISPATCH', None)
    if args.mode is not None:
        run_precision(mode=args.mode, output=args.output)
        return
    for mode in ['fp32', 'tf32', 'tf32x3']:
        with (args.output / f'{mode}.log').open('w') as log:
            subprocess.run(
                args=[sys.executable, __file__, '--output', str(args.output), '--mode', mode],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    result = {mode: json.loads((args.output / f'{mode}.json').read_text()) for mode in ['fp32', 'tf32', 'tf32x3']}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
