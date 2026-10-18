# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Console tables and regression summaries for benchmark results."""

from __future__ import annotations

from typing import Any

_RESULT_RESERVED_KEYS = {'op', 'mode', 'B', 'T', 'H', 'D', 'median_ms', 'p20_ms', 'p80_ms'}


def format_machine_info(info: dict) -> str:
    gpu = info.get('gpu_name', 'N/A')
    pytorch = info.get('pytorch_version', 'N/A')
    platform = info.get('device_platform', 'N/A')
    if platform == 'cuda':
        backend = f"CUDA {info.get('cuda_version', 'N/A')}"
    elif platform == 'npu':
        backend = f"NPU: CANN {info.get('cann_version', 'N/A')}"
    else:
        backend = platform.upper() if platform != 'N/A' else 'N/A'
    return f"Machine: {gpu} | {backend} | PyTorch {pytorch}"


def _make_result_key(r):
    """Identify a result row. Include any extra shape dims (e.g., AttnRes `L`)
    so per-op shape configs that share B/T/H/D but vary in extras don't collide.
    """
    extras = tuple(sorted((k, v) for k, v in r.items() if k not in _RESULT_RESERVED_KEYS))
    return (r['op'], r['mode'], r['B'], r['T'], r['H'], r['D'], extras)


def _truncate_branch(name: str, max_len: int = 8) -> str:
    """Truncate a branch name to *max_len* chars, adding ``...`` if needed."""
    if len(name) > max_len:
        return name[:max_len] + '...'
    return name


def _make_col_headers(old_git: str, new_git: str) -> tuple[str, str]:
    """Build equal-width column headers like ``main    [abc1234](ms)``.

    Both branch names are truncated to 8 chars (with ``...``), then
    right-padded to the same width so the ``[commit](ms)`` parts align.
    """
    def _branch(git_label):
        return git_label.split('[')[0] if '[' in git_label else git_label

    def _suffix(git_label):
        if '[' in git_label:
            return '[' + git_label.split('[', 1)[1] + '(ms)'
        return '(ms)'

    old_br = _truncate_branch(_branch(old_git))
    new_br = _truncate_branch(_branch(new_git))
    br_w = max(len(old_br), len(new_br))
    return (f"{old_br:>{br_w}s}{_suffix(old_git)}",
            f"{new_br:>{br_w}s}{_suffix(new_git)}")


def print_results_table(
    results: list[dict],
    machine_info: dict | None = None,
    baseline: list[dict] | None = None,
    baseline_info: dict | None = None,
):
    """Print old / new / speedup comparison table.

    Column order: ``mode  B  T  H  D  op  old(ms)  new(ms)  speedup``.
    The mode column (``fwd`` / ``fwdbwd``) is shown when it changes.
    A dash separator (not covering the mode column) and column header
    are repeated before each shape group.
    """
    if not results:
        print("\n  No results to display.")
        return

    has_baseline = baseline is not None and len(baseline) > 0
    old_map = {_make_result_key(r): r for r in baseline} if has_baseline else {}

    new_git = machine_info.get('git_label', 'new') if machine_info else 'new'

    mode_pad = ' ' * 10

    # mixed workloads omit L because it only describes layer-axis aggregation.
    has_l = bool(results) and all('L' in r for r in results)
    l_col = f"{'L':>4s} " if has_l else ''
    l_extra_w = 5 if has_l else 0

    if has_baseline:
        old_git = baseline_info.get('git_label', 'main') if baseline_info else 'main'
        old_hdr, new_hdr = _make_col_headers(old_git, new_git)
        col_w = max(len(old_hdr), len(new_hdr), 10)
        inner_w = l_extra_w + 4 + 1 + 6 + 1 + 4 + 1 + 4 + 2 + 28 + 2 + col_w + 1 + col_w + 1 + 8
        inner_hdr = (f"{l_col}{'B':>4s} {'T':>6s} {'H':>4s} {'D':>4s}  {'op':<28s}"
                     f"  {old_hdr:>{col_w}s} {new_hdr:>{col_w}s} {'speedup':>8s}")
    else:
        new_hdr = _truncate_branch(new_git.split('[')[0]) if '[' in new_git else new_git
        suffix = '[' + new_git.split('[', 1)[1] + '(ms)' if '[' in new_git else '(ms)'
        new_hdr = new_hdr + suffix
        col_w = max(len(new_hdr), 10)
        inner_w = l_extra_w + 4 + 1 + 6 + 1 + 4 + 1 + 4 + 2 + 28 + 2 + col_w
        inner_hdr = (f"{l_col}{'B':>4s} {'T':>6s} {'H':>4s} {'D':>4s}  {'op':<28s}"
                     f"  {new_hdr:>{col_w}s}")

    width = 10 + inner_w
    sep = '=' * width
    dash_line = mode_pad + '-' * inner_w

    print(f"\n{sep}")
    if machine_info:
        print(f"  {format_machine_info(machine_info)}")

    def _l_cell(r, blank=False):
        if not has_l:
            return ''
        if blank:
            return f"{'':>4s} "
        v = r.get('L')
        return f"{v:>4d} " if v is not None else f"{'-':>4s} "

    prev_shape = None
    prev_mode = None
    for r in results:
        cur_shape = (r.get('L'), r['B'], r['T'], r['H'], r['D'])
        cur_mode = r['mode']

        if cur_mode != prev_mode:
            print(sep)
            print(f"  {cur_mode:<7s} {inner_hdr}")
            print(dash_line)
        elif cur_shape != prev_shape:
            print(dash_line)

        if cur_mode != prev_mode or cur_shape != prev_shape:
            shape_str = f"{_l_cell(r)}{r['B']:>4d} {r['T']:>6d} {r['H']:>4d} {r['D']:>4d}"
        else:
            shape_str = f"{_l_cell(r, blank=True)}{'':>4s} {'':>6s} {'':>4s} {'':>4s}"

        prev_shape = cur_shape
        prev_mode = cur_mode

        new_ms = r['median_ms']
        prefix = f"{mode_pad}{shape_str}  {r['op']:<28s}"
        if has_baseline:
            old_r = old_map.get(_make_result_key(r))
            if old_r:
                old_ms = old_r['median_ms']
                speedup = old_ms / new_ms if new_ms > 0 else float('inf')
                print(f"{prefix}  {old_ms:>{col_w}.3f} {new_ms:>{col_w}.3f} {speedup:>7.2f}x")
            else:
                print(f"{prefix}  {'-':>{col_w}s} {new_ms:>{col_w}.3f} {'-':>8s}")
        else:
            print(f"{prefix}  {new_ms:>{col_w}.3f}")

    print(sep)


def _result_key(result: dict[str, Any]) -> tuple[str, str, str]:
    return result["op"], result["shape"], result["mode"]


def _format_latency(metrics: dict[str, float] | None, statistic: str) -> str:
    return "-" if metrics is None else f"{metrics[f'{statistic}_ms']:.3f}"


def _format_config(row: dict[str, Any]) -> str:
    """Format extra configuration, omitting dimensions already shown in the table."""

    excluded = {
        "op", "shape", "mode", "B", "T", "H", "D", "N",
        "eager", "graph", "capture_ms", "validation",
    }
    preferred = ("dtype", "chunk_size", "actual_T", "HV", "DV")
    for name, column in (("actual_T", "T"), ("HV", "H"), ("DV", "D")):
        if name in row and column in row and row[name] == row[column]:
            excluded.add(name)
    parts = [f"{name}={row[name]}" for name in preferred if name in row and name not in excluded]
    for name in sorted(set(row) - excluded - set(preferred)):
        value = row[name]
        if value is False or value in (None, "balanced"):
            continue
        parts.append(f"{name}={value}")
    return ",".join(parts) or "-"


def print_graph_results(current, current_info, baseline=None, baseline_info=None) -> None:
    baseline_map = {_result_key(row): row for row in baseline or []}
    old_label = baseline_info["git_label"] if baseline_info else "base"
    new_label = current_info["git_label"]
    title = f"{old_label} eager / {new_label} eager / {new_label} graph (ms)"
    print(f"\n{title}")
    print(f"Device: {current_info['device']} | torch {current_info['torch']} | torch_npu {current_info['torch_npu']}")
    header = (
        f"{'mode':<8} {'case':<24} {'stat':<5} {'B':>3} {'T':>6} {'H':>4} {'D':>4} {'N':>4} "
        f"{'base eager':>12} {'HEAD eager':>12} {'HEAD graph':>12} {'eager x':>9} {'graph x':>9}    "
        f"config"
    )
    print(header)
    print("-" * len(header))
    for row in current:
        old = baseline_map.get(_result_key(row))
        for statistic in ("p50", "p95"):
            base_metrics = old.get("eager") if old else None
            eager_metrics = row.get("eager")
            graph_metrics = row.get("graph")
            eager_speedup = "-"
            graph_speedup = "-"
            if base_metrics and eager_metrics:
                eager_speedup = f"{base_metrics[f'{statistic}_ms'] / eager_metrics[f'{statistic}_ms']:.2f}x"
            if eager_metrics and graph_metrics:
                graph_speedup = f"{eager_metrics[f'{statistic}_ms'] / graph_metrics[f'{statistic}_ms']:.2f}x"
            print(
                f"{row['mode']:<8} {row['shape']:<24} {statistic:<5} "
                f"{row['B']:>3} {row['T']:>6} {row['H']:>4} {row['D']:>4} {row.get('N', row['B']):>4} "
                f"{_format_latency(base_metrics, statistic):>12} "
                f"{_format_latency(eager_metrics, statistic):>12} "
                f"{_format_latency(graph_metrics, statistic):>12} "
                f"{eager_speedup:>9} {graph_speedup:>9}    "
                f"{_format_config(row)}"
            )


def _truncate(s: str, max_len: int = 8) -> str:
    """Truncate *s* to *max_len* chars, adding ``...`` if needed."""
    return s if len(s) <= max_len else s[:max_len] + '...'


def print_comparison(
    base_results: list[dict],
    head_results: list[dict],
    base_sha: str,
    head_sha: str,
    threshold: float,
    machine_info: dict | None = None,
) -> tuple[int, list, list]:
    """Print latency changes and return the exit code, regressions, and speedups."""

    def make_key(r):
        return (r['op'], r['mode'], r['B'], r['T'], r['H'], r['D'])

    head_map = {make_key(r): r for r in head_results}
    base_map = {make_key(r): r for r in base_results}

    mode_order = {'fwd': 0, 'fwdbwd': 1}
    all_keys = sorted(
        set(list(head_map.keys()) + list(base_map.keys())),
        key=lambda k: (mode_order.get(k[1], 9), k[0], k[2], k[3], k[4], k[5]),
    )

    base_hdr = f"base:{_truncate(base_sha)}(ms)"
    head_hdr = f"head:{_truncate(head_sha)}(ms)"
    col_w = max(len(base_hdr), len(head_hdr), 10)

    width = 2 + 28 + 1 + 7 + 1 + 4 + 1 + 6 + 1 + 4 + 1 + 4 + 2 + col_w + 1 + col_w + 1 + 8 + 1 + 8

    print(f"\n{'=' * width}")
    if machine_info:
        gpu = machine_info.get('gpu_name', 'N/A')
        cuda = machine_info.get('cuda_version', 'N/A')
        pytorch = machine_info.get('pytorch_version', 'N/A')
        print(f"  Machine: {gpu} | CUDA {cuda} | PyTorch {pytorch}")
    print(f"{'=' * width}")
    print(f"  {'op':<28s} {'mode':<7s} {'B':>4s} {'T':>6s} {'H':>4s} {'D':>4s}"
          f"  {base_hdr:>{col_w}s} {head_hdr:>{col_w}s} {'speedup':>8s} {'change':>8s}")
    print(f"  {'-' * 28} {'-' * 7} {'-' * 4} {'-' * 6} {'-' * 4} {'-' * 4}"
          f"  {'-' * col_w} {'-' * col_w} {'-' * 8} {'-' * 8}")

    regressions = []
    speedups = []
    prev_mode = None
    for key in all_keys:
        op, mode, B, T, H, D = key

        if prev_mode is not None and mode != prev_mode:
            print()
        prev_mode = mode

        base_r = base_map.get(key)
        head_r = head_map.get(key)

        prefix = f"  {op:<28s} {mode:<7s} {B:>4d} {T:>6d} {H:>4d} {D:>4d}"
        if base_r and head_r:
            base_ms = base_r['median_ms']
            head_ms = head_r['median_ms']
            change_pct = (head_ms - base_ms) / base_ms * 100
            speedup = base_ms / head_ms if head_ms > 0 else float('inf')
            sign = '+' if change_pct > 0 else ''
            marker = ''
            if change_pct > threshold:
                marker = ' 🔴'
            elif change_pct < -threshold:
                marker = ' 🟢'
            else:
                marker = ''
            print(f"{prefix}  {base_ms:>{col_w}.3f} {head_ms:>{col_w}.3f} {speedup:>7.2f}x "
                  f"{sign}{change_pct:>6.1f}%{marker}")
            if change_pct > threshold:
                regressions.append((key, base_ms, head_ms, change_pct))
            elif change_pct < -threshold:
                speedups.append((key, base_ms, head_ms, change_pct))
        elif head_r:
            print(f"{prefix}  {'-':>{col_w}s} {head_r['median_ms']:>{col_w}.3f} {'':>8s} {'new':>8s}")
        elif base_r:
            print(f"{prefix}  {base_r['median_ms']:>{col_w}.3f} {'-':>{col_w}s} {'':>8s} {'removed':>8s}")

    print(f"{'=' * width}")

    if regressions:
        print(f"\n  WARNING: {len(regressions)} regression(s) detected (>{threshold}% slower):")
        for key, base_ms, head_ms, pct in regressions:
            print(f"    {key}: {base_ms:.3f} -> {head_ms:.3f} ms (+{pct:.1f}%)")

    if speedups:
        print(f"\n  INFO: {len(speedups)} speedup(s) detected (<-{threshold}% faster):")
        for key, base_ms, head_ms, pct in speedups:
            print(f"    {key}: {base_ms:.3f} -> {head_ms:.3f} ms ({pct:.1f}%)")

    if not regressions:
        print(f"\n  No regressions detected (threshold: {threshold}%).")

    return (1 if regressions else 0, regressions, speedups)
