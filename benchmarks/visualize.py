# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Visualization helpers for FLA benchmark outputs.

Supports:
  - Triton ``perf_report`` CSV files (module benchmarks)
  - ``benchmark_results.json`` from ``scripts/run_benchmark_compare.py``

Usage::

    # From a module benchmark script (recommended)
    python benchmarks/modules/benchmark_activations.py

    # After running a module benchmark that writes CSV via save_path
    python -m benchmarks.visualize perf-report activation_benchmark/activation_performance.csv

    # From a CI / local comparison JSON
    python -m benchmarks.visualize comparison benchmark_results.json --output plots/
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

_SHAPE_COLS = frozenset({'B', 'T', 'H', 'D', 'L'})


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for benchmark visualization. "
            "Install with: pip install '.[benchmark]'"
        ) from exc
    return plt


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for benchmark visualization. "
            "Install with: pip install pandas"
        ) from exc
    return pd


def _split_shape_and_metric_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    shape_cols = [c for c in columns if c in _SHAPE_COLS]
    metric_cols = [c for c in columns if c not in shape_cols]
    return shape_cols, metric_cols


def _pick_x_axis(shape_cols: list[str], df) -> str:
    if not shape_cols:
        raise ValueError("No shape columns found in CSV")

    preferred = ['T', 'D', 'H', 'B', 'L']
    candidates = sorted(
        shape_cols,
        key=lambda c: (
            -(df[c].nunique() if c in df.columns else 0),
            preferred.index(c) if c in preferred else len(preferred),
        ),
    )
    x_axis = candidates[0]
    if df[x_axis].nunique() <= 1:
        raise ValueError(
            f"Cannot pick a varying x-axis from shape columns {shape_cols}"
        )
    return x_axis


def _facet_columns(shape_cols: list[str], x_axis: str, df) -> list[str]:
    facets = []
    for col in shape_cols:
        if col == x_axis:
            continue
        if df[col].nunique() > 1:
            facets.append(col)
    return facets


def _iter_facet_groups(df, facet_cols: list[str]):
    """Yield (mask, title) for each facet group present in the data."""
    if not facet_cols:
        yield df.index, ''
        return

    grouped = df.groupby(facet_cols, sort=True)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        title = ', '.join(f'{col}={int(val) if val == int(val) else val}'
                          for col, val in zip(facet_cols, keys, strict=True))
        yield sub.index, title


def _split_metric_groups(metric_cols: list[str]) -> tuple[list[str], list[str]]:
    """Split providers into forward and backward groups when both exist."""
    bwd_cols = [m for m in metric_cols if m.endswith('_fwdbwd') or m.endswith('_bwd')]
    fwd_cols = [m for m in metric_cols if m not in bwd_cols]
    if fwd_cols and bwd_cols:
        return fwd_cols, bwd_cols
    return metric_cols, []


def _grid_shape(n_facets: int, n_mode_rows: int) -> tuple[int, int, int]:
    """Pick a wide-friendly subplot grid.

    Returns (nrows, ncols, rows_per_mode). For single-mode layouts rows_per_mode
    equals nrows; for forward/backward stacks each mode occupies rows_per_mode rows.
    """
    if n_facets <= 1:
        ncols = 1
    elif n_mode_rows > 1:
        ncols = min(n_facets, 5)
    elif n_facets <= 3:
        ncols = n_facets
    else:
        ncols = min(3, n_facets)

    rows_per_mode = math.ceil(n_facets / ncols) if n_facets else 1
    nrows = n_mode_rows * rows_per_mode
    return nrows, ncols, rows_per_mode


def _provider_style(name: str) -> tuple[str, str]:
    if name.endswith('_fwdbwd'):
        return name.removesuffix('_fwdbwd'), '--'
    if name.endswith('_fwd'):
        return name.removesuffix('_fwd'), '-'
    if name.endswith('_bwd'):
        return name.removesuffix('_bwd'), '--'
    return name, '-'


def visualize_perf_report_csv(
    csv_path: str | os.PathLike,
    save_dir: str | os.PathLike | None = None,
    *,
    x_axis: str | None = None,
    title: str | None = None,
) -> list[str]:
    """Plot a Triton perf_report CSV with sensible axes for multi-dim sweeps.

    When the CSV has multiple shape columns (e.g. B, T, D) but only one varies
    meaningfully on the x-axis, this creates one subplot per facet combination
    (e.g. one panel per D) instead of plotting the constant B column.

    Returns a list of saved image paths.
    """
    pd = _require_pandas()
    plt = _require_matplotlib()

    csv_path = Path(csv_path)
    if save_dir is None:
        save_dir = csv_path.parent
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    shape_cols, metric_cols = _split_shape_and_metric_columns(df.columns.tolist())
    if not metric_cols:
        raise ValueError(f"No metric columns found in {csv_path}")

    if x_axis is None:
        x_axis = _pick_x_axis(shape_cols, df)
    elif x_axis not in shape_cols:
        raise ValueError(f"x_axis '{x_axis}' is not among shape columns {shape_cols}")

    facet_cols = _facet_columns(shape_cols, x_axis, df)
    fixed_cols = [c for c in shape_cols if c not in facet_cols and c != x_axis]

    facet_groups = list(_iter_facet_groups(df, facet_cols))
    n_facets = len(facet_groups)

    fwd_cols, bwd_cols = _split_metric_groups(metric_cols)
    mode_groups = [('Forward', fwd_cols)]
    if bwd_cols:
        mode_groups.append(('Backward', bwd_cols))

    n_mode_rows = len(mode_groups)
    nrows, ncols, rows_per_mode = _grid_shape(n_facets, n_mode_rows)
    fig_w = max(3.0 * ncols, 8)
    fig_h = max(3.8 * nrows, 4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    palette = plt.get_cmap('tab10')
    color_map: dict[str, tuple] = {}
    legend_handles: dict[str, object] = {}

    for mode_row, (mode_label, cols) in enumerate(mode_groups):
        for facet_idx, (facet_index, facet_title) in enumerate(facet_groups):
            row = mode_row * rows_per_mode + facet_idx // ncols
            col = facet_idx % ncols
            ax = axes[row, col]

            sub = df.loc[facet_index].sort_values(x_axis)
            if sub.empty:
                ax.set_visible(False)
                continue

            for metric in cols:
                base, linestyle = _provider_style(metric)
                if base not in color_map:
                    color_map[base] = palette(len(color_map) % 10)
                line, = ax.plot(
                    sub[x_axis],
                    sub[metric],
                    label=base,
                    color=color_map[base],
                    linestyle=linestyle,
                    marker='o',
                    markersize=3,
                )
                legend_handles[base] = line

            # Column header on the top row of each mode block only.
            if facet_idx // ncols == 0:
                title_parts = []
                if facet_title:
                    title_parts.append(facet_title)
                for col_name in fixed_cols:
                    vals = sorted(df[col_name].unique())
                    if len(vals) == 1:
                        val = vals[0]
                        val_str = int(val) if val == int(val) else val
                        title_parts.append(f'{col_name}={val_str}')
                if title_parts:
                    ax.set_title(', '.join(title_parts), fontsize=9, pad=8)

            if facet_idx % ncols == 0:
                if n_mode_rows > 1:
                    ax.set_ylabel(f'{mode_label}\nTime (ms)', fontsize=9)
                else:
                    ax.set_ylabel('Time (ms)', fontsize=9)

            ax.set_xlabel(x_axis, fontsize=9)

            ax.grid(True, alpha=0.3)

        # Hide unused facet slots in this mode block.
        for empty_idx in range(n_facets, rows_per_mode * ncols):
            row = mode_row * rows_per_mode + empty_idx // ncols
            col = empty_idx % ncols
            axes[row, col].set_visible(False)

    plot_title = title or csv_path.stem.replace('_', ' ').title()
    fig.suptitle(plot_title, y=0.98, fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.14, wspace=0.22, hspace=0.35)

    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc='lower center',
            ncol=min(len(legend_handles), 6),
            fontsize=8,
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )

    out_path = save_dir / f"{csv_path.stem}_faceted.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)

    html_path = save_dir / 'results.html'
    _write_html_gallery(html_path, [out_path], title=plot_title)

    return [str(out_path)]


def _make_comparison_label(r: dict) -> str:
    parts = [r['op'], r['mode'], f"B{r['B']}", f"T{r['T']}", f"H{r['H']}", f"D{r['D']}"]
    if 'L' in r:
        parts.append(f"L{r['L']}")
    return ' / '.join(parts)


def visualize_comparison(
    json_path: str | os.PathLike,
    save_dir: str | os.PathLike | None = None,
    *,
    threshold: float | None = None,
    top_n: int = 20,
) -> list[str]:
    """Plot base vs head latency and percent-change from a comparison JSON."""
    plt = _require_matplotlib()

    json_path = Path(json_path)
    if save_dir is None:
        save_dir = json_path.parent / 'plots'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    base_results = data.get('base_results', [])
    head_results = data.get('head_results', [])
    threshold = threshold if threshold is not None else 5.0

    def _key(r):
        extras = tuple(sorted((k, v) for k, v in r.items()
                              if k not in {'op', 'mode', 'B', 'T', 'H', 'D',
                                           'median_ms', 'p20_ms', 'p80_ms'}))
        return (r['op'], r['mode'], r['B'], r['T'], r['H'], r['D'], extras)

    base_map = {_key(r): r for r in base_results}
    head_map = {_key(r): r for r in head_results}
    all_keys = sorted(set(base_map) | set(head_map))

    rows = []
    for key in all_keys:
        b = base_map.get(key)
        h = head_map.get(key)
        if b and h:
            base_ms = b['median_ms']
            head_ms = h['median_ms']
            change_pct = (head_ms - base_ms) / base_ms * 100 if base_ms else 0.0
            rows.append({
                'label': _make_comparison_label(b),
                'base_ms': base_ms,
                'head_ms': head_ms,
                'change_pct': change_pct,
            })

    if not rows:
        return []

    rows.sort(key=lambda r: abs(r['change_pct']), reverse=True)
    rows = rows[:top_n]

    saved: list[str] = []

    # Percent-change bar chart
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(rows))))
    labels = [r['label'] for r in rows][::-1]
    changes = [r['change_pct'] for r in rows][::-1]
    colors = []
    for c in changes:
        if c > threshold:
            colors.append('#d62728')
        elif c < -threshold:
            colors.append('#2ca02c')
        else:
            colors.append('#7f7f7f')
    ax.barh(labels, changes, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(threshold, color='#d62728', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axvline(-threshold, color='#2ca02c', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Change (%)')
    base_sha = data.get('base_sha', 'base')
    head_sha = data.get('head_sha', 'head')
    ax.set_title(f'Benchmark Change: {base_sha} → {head_sha}')
    fig.tight_layout()
    change_path = save_dir / 'comparison_change_pct.png'
    fig.savefig(change_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(str(change_path))

    # Grouped latency bar chart
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(rows))))
    y_pos = list(range(len(rows)))
    height = 0.35
    base_vals = [r['base_ms'] for r in rows][::-1]
    head_vals = [r['head_ms'] for r in rows][::-1]
    ax.barh([y - height / 2 for y in y_pos], base_vals, height=height,
            label=f"base ({base_sha})", color='#1f77b4')
    ax.barh([y + height / 2 for y in y_pos], head_vals, height=height,
            label=f"head ({head_sha})", color='#ff7f0e')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Median latency (ms)')
    ax.set_title('Base vs Head Latency')
    ax.legend()
    fig.tight_layout()
    latency_path = save_dir / 'comparison_latency.png'
    fig.savefig(latency_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(str(latency_path))

    _write_html_gallery(
        save_dir / 'results.html',
        [Path(p) for p in saved],
        title=f'Benchmark Comparison {base_sha} → {head_sha}',
    )
    return saved


def _write_html_gallery(html_path: Path, image_paths: list[Path], *, title: str):
    lines = [
        '<html><head>',
        f'<title>{title}</title>',
        '<style>body{font-family:sans-serif;margin:24px;} img{max-width:100%;margin:16px 0;}</style>',
        '</head><body>',
        f'<h1>{title}</h1>',
    ]
    for img in image_paths:
        rel = os.path.relpath(img, html_path.parent)
        lines.append(f'<img src="{rel}" alt="{img.stem}"/>')
    lines.append('</body></html>')
    html_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Visualize FLA benchmark results')
    sub = parser.add_subparsers(dest='command', required=True)

    perf = sub.add_parser('perf-report', help='Visualize a Triton perf_report CSV')
    perf.add_argument('csv', help='Path to the CSV file')
    perf.add_argument('--output', '-o', default=None, help='Output directory for plots')
    perf.add_argument('--x-axis', default=None, help='Shape column for the x-axis (default: auto)')

    cmp = sub.add_parser('comparison', help='Visualize a benchmark_results.json comparison')
    cmp.add_argument('json', help='Path to benchmark_results.json')
    cmp.add_argument('--output', '-o', default=None, help='Output directory for plots')
    cmp.add_argument('--threshold', type=float, default=None, help='Regression threshold (%%)')
    cmp.add_argument('--top-n', type=int, default=20, help='Max rows to plot')

    args = parser.parse_args()

    if args.command == 'perf-report':
        paths = visualize_perf_report_csv(
            args.csv,
            save_dir=args.output,
            x_axis=args.x_axis,
        )
    else:
        paths = visualize_comparison(
            args.json,
            save_dir=args.output,
            threshold=args.threshold,
            top_n=args.top_n,
        )

    for p in paths:
        print(f"Saved: {p}")


if __name__ == '__main__':
    main()
