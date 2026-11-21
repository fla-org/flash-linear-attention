# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""For cold comparisons, run each revision/order with a fresh TRITON_CACHE_DIR and FLA_CACHE_RESULTS=0."""

import argparse
import json
import time

import torch
import triton
import triton.testing

from fla.modules.norm.fused_norm_gate import layer_norm_gated_bwd, layer_norm_gated_fwd


def main():
    parser = argparse.ArgumentParser(description="Measure cold length sweeps and warm gated RMSNorm calls.")
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--lengths', type=int, default=16)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(42)
    lengths = [index * 65536 + 1 for index in range(args.lengths)]
    if args.reverse:
        lengths.reverse()
    results = []
    for rows in lengths:
        x = torch.randn(rows, args.dim, device='cuda', dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        dy = torch.randn_like(x)
        weight = torch.randn(args.dim, device='cuda', dtype=torch.bfloat16)

        def call():
            y, mean, rstd, residual = layer_norm_gated_fwd(x=x, g=gate, weight=weight, bias=None, is_rms_norm=True)
            grads = layer_norm_gated_bwd(
                dy=dy,
                x=residual,
                g=gate,
                weight=weight,
                bias=None,
                mean=mean,
                rstd=rstd,
                is_rms_norm=True,
            )
            return y, grads

        torch.cuda.synchronize()
        start = time.perf_counter()
        y, grads = call()
        torch.cuda.synchronize()
        cold_seconds = time.perf_counter() - start
        assert torch.isfinite(y).all()
        assert all(grad is None or torch.isfinite(grad).all() for grad in grads)
        del y, grads
        median, low, high = triton.testing.do_bench(fn=call, warmup=50, rep=200, quantiles=[0.5, 0.2, 0.8])
        row = dict(rows=rows, cold_seconds=cold_seconds, warm_ms=median, warm_p20_ms=low, warm_p80_ms=high)
        results.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps(dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        dim=args.dim,
        reverse=args.reverse,
        cold_total_seconds=sum(row['cold_seconds'] for row in results),
        results=results,
    )), flush=True)


if __name__ == '__main__':
    main()
