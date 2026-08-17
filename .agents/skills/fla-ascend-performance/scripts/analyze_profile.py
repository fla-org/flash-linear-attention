# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Generic Ascend profiler CSV analyzer.

Parses ASCEND_PROFILER_OUTPUT under a trace directory and prints:
- op_statistic top kernels by total time
- kernel_details pipe / UB / bandwidth columns (whatever the run collected)

Op-agnostic: no fla imports. Filter by kernel name substring when needed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections.abc import Iterable


def find_csv(trace_dir: str, filename: str) -> str | None:
    paths = glob.glob(os.path.join(trace_dir, "**", filename), recursive=True)
    return paths[0] if paths else None


def _pick_cols(keys: Iterable[str], *needles: str) -> list[str]:
    out = []
    for c in keys:
        cl = c.lower()
        if any(n in cl for n in needles):
            out.append(c)
    return out


def _nonzero_vals(row: dict, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        v = row.get(c, "")
        if v not in (None, "", "N/A"):
            out[c] = v
    return out


def summarize_op_statistic(trace_dir: str, top_k: int = 15) -> None:
    path = find_csv(trace_dir, "op_statistic.csv")
    if not path:
        print(f"[analyze] no op_statistic.csv under {trace_dir}")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[analyze] empty op_statistic.csv: {path}")
        return

    def _total(r):
        try:
            return float(r.get("Total Time(us)", 0) or 0)
        except ValueError:
            return 0.0

    rows = sorted(rows, key=_total, reverse=True)
    print(f"\n=== op_statistic top {top_k} ({path}) ===")
    print(f"{'OP Type':<48} {'Core':<16} {'Count':>6} {'Total(us)':>12} {'Ratio%':>8}")
    for r in rows[:top_k]:
        print(
            f"{r.get('OP Type', '?'):<48} "
            f"{r.get('Core Type', '?'):<16} "
            f"{r.get('Count', '?'):>6} "
            f"{r.get('Total Time(us)', '?'):>12} "
            f"{r.get('Ratio(%)', '?'):>8}"
        )


def summarize_kernel_details(
    trace_dir: str,
    kernel_filter: str | None = None,
    top_k: int = 20,
) -> None:
    path = find_csv(trace_dir, "kernel_details.csv")
    if not path:
        print(f"[analyze] no kernel_details.csv under {trace_dir}")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[analyze] empty kernel_details.csv: {path}")
        return

    keys = list(rows[0].keys())
    if kernel_filter:
        targets = [r for r in rows if kernel_filter in r.get("Name", "")]
        if not targets:
            print(f"[analyze] no kernel Name containing {kernel_filter!r}; showing top by duration")
            targets = rows
    else:
        targets = rows

    def _dur(r):
        try:
            return float(r.get("Duration(us)", 0) or 0)
        except ValueError:
            return 0.0

    targets = sorted(targets, key=_dur, reverse=True)[:top_k]

    # PipeUtilization groups
    groups = {
        "Cube(MAC)": _pick_cols(keys, "mac_ratio", "cube_utilization"),
        "Vector": _pick_cols(keys, "vec_ratio"),
        "Scalar": _pick_cols(keys, "scalar_ratio", "scalar_time"),
        "MTE1": _pick_cols(keys, "mte1"),
        "MTE2": _pick_cols(keys, "mte2"),
        "MTE3": _pick_cols(keys, "mte3"),
        "Fixpipe": _pick_cols(keys, "fixpipe"),
        "UB_bw": _pick_cols(keys, "ub_"),
        "L2/cache": _pick_cols(keys, "l2", "icache"),
    }
    # Drop empty groups so output matches the metrics that were actually collected.
    groups = {k: v for k, v in groups.items() if v}

    print(f"\n=== kernel_details ({path}) ===")
    print(f"columns present: {sorted({c for cols in groups.values() for c in cols})}")
    if not groups:
        print("  (no pipe/UB metric columns; re-profile with --metrics PipeUtilization or MemoryUB)")
        print(f"  available columns: {keys}")
        return

    for r in targets:
        name = r.get("Name", "?")
        dur = r.get("Duration(us)", "")
        core = r.get("Accelerator Core", r.get("Type", ""))
        print(f"\n  {name}  Duration={dur} us  core={core}")
        for label, cols in groups.items():
            vals = _nonzero_vals(r, cols)
            if vals:
                print(f"    {label}: {vals}")


def summarize_trace(
    trace_dir: str,
    kernel_filter: str | None = None,
    top_k: int = 20,
) -> None:
    """Print op_statistic + kernel_details summary for a profiling run."""
    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(trace_dir)
    summarize_op_statistic(trace_dir, top_k=min(top_k, 15))
    summarize_kernel_details(trace_dir, kernel_filter=kernel_filter, top_k=top_k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Ascend profiler CSV outputs")
    parser.add_argument("trace_dir", help="Profiling output directory (or parent containing ASCEND_PROFILER_OUTPUT)")
    parser.add_argument("--kernel-filter", default=None, help="Substring match on kernel Name")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    summarize_trace(args.trace_dir, kernel_filter=args.kernel_filter, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
