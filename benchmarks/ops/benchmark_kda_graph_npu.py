# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compatibility entry point for the KDA Ascend NPUGraph benchmark.

New benchmark automation should use ``python -m benchmarks.ops.run_graph``.
This wrapper preserves the original KDA-specific command line and JSON shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_graph import benchmark_graph_op  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--value-heads", type=int)
    parser.add_argument("--value-dim", type=int)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--chunk-size", choices=(32, 64), type=int, default=64)
    parser.add_argument("--fused-options", action="store_true")
    parser.add_argument("--safe-gate", action="store_true")
    parser.add_argument("--allow-neg-eigval", action="store_true")
    parser.add_argument("--disable-recompute", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    shape = {
        "B": 1,
        "T": args.tokens,
        "H": args.heads,
        "D": args.dim,
        "N": args.num_seqs,
        "HV": args.value_heads or args.heads,
        "DV": args.value_dim or args.dim,
        "dtype": args.dtype,
        "chunk_size": args.chunk_size,
        "fused_options": args.fused_options,
        "safe_gate": args.safe_gate,
        "allow_neg_eigval": args.allow_neg_eigval,
        "disable_recompute": args.disable_recompute,
    }
    rows = benchmark_graph_op(
        op_name="chunk_kda",
        shapes={"legacy": shape},
        modes=("fwdbwd",),
        engines=("eager", "graph"),
        default_dtype=args.dtype,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=0,
        progress=False,
    )
    row = rows[0]

    import torch
    import torch_npu

    result = {
        "config": {
            "tokens": args.tokens,
            "heads": args.heads,
            "value_heads": shape["HV"],
            "key_dim": args.dim,
            "value_dim": shape["DV"],
            "num_seqs": args.num_seqs,
            "dtype": str(getattr(torch, args.dtype)),
            "chunk_size": args.chunk_size,
            "fused_options": args.fused_options,
            "safe_gate": args.safe_gate,
            "allow_neg_eigval": args.allow_neg_eigval,
            "disable_recompute": args.disable_recompute,
            "device": torch.npu.get_device_name(torch.npu.current_device()),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "capture_ms": row["capture_ms"],
        "validation": row["validation"],
        "eager_live": row["eager"],
        "graph_update_replay": row["graph"],
        "p50_speedup": row["eager"]["p50_ms"] / row["graph"]["p50_ms"],
        "p95_speedup": row["eager"]["p95_ms"] / row["graph"]["p95_ms"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
