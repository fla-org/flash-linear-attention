# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared runner for Triton ``perf_report`` module benchmarks."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def ensure_repo_root_on_path() -> Path:
    """Make ``benchmarks.*`` importable when scripts are run directly."""
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def default_save_dir(script_file: str) -> str:
    """Derive ``./<module>_benchmark`` from ``benchmark_<module>.py``."""
    stem = Path(script_file).stem
    name = stem.removeprefix('benchmark_')
    return f'./{name}_benchmark'


def get_plot_name(mark: Any) -> str:
    benches = mark.benchmarks
    if not isinstance(benches, list):
        benches = [benches]
    return benches[0].plot_name


def run_module_benchmark(
    mark: Any,
    *,
    save_dir: str | None = None,
    plot_name: str | None = None,
    script_file: str | None = None,
    print_data: bool = True,
    visualize: bool = True,
    **run_kwargs: Any,
) -> str:
    """Run a module benchmark and optionally generate faceted plots.

    Args:
        mark: Object returned by ``@triton.testing.perf_report``.
        save_dir: Output directory for CSV/PNG/HTML. Defaults to
            ``./<module>_benchmark`` inferred from *script_file*.
        plot_name: CSV/plot stem from the Benchmark config. Auto-detected when
            omitted.
        script_file: Pass ``__file__`` to enable the default *save_dir*.
        print_data: Print the benchmark table to stdout.
        visualize: Generate faceted plots via ``benchmarks.visualize``.
        **run_kwargs: Extra kwargs forwarded to ``mark.run``.

    Returns:
        The output directory path.
    """
    ensure_repo_root_on_path()

    if plot_name is None:
        plot_name = get_plot_name(mark)
    if save_dir is None:
        if script_file is None:
            raise ValueError("save_dir or script_file must be provided")
        save_dir = default_save_dir(script_file)

    save_dir = str(Path(save_dir).resolve())
    os.environ.setdefault('MPLBACKEND', 'Agg')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    mark.run(print_data=print_data, save_path=save_dir, **run_kwargs)

    if visualize:
        csv_path = Path(save_dir) / f'{plot_name}.csv'
        if csv_path.exists():
            try:
                from benchmarks.visualize import visualize_perf_report_csv
                for path in visualize_perf_report_csv(csv_path, save_dir=save_dir):
                    print(f"Saved plot: {path}")
            except ImportError as exc:
                print(
                    f"Warning: skipping visualization (install matplotlib and pandas: {exc})",
                    file=sys.stderr,
                )
        else:
            print(f"Warning: expected CSV not found at {csv_path}, skipping visualization")

    return save_dir
