# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Execute Triton provider sweeps and return the common benchmark result format."""

import os
from pathlib import Path

import torch

from benchmarks.utils.reporting import BenchmarkResult


def add_module_arguments(parser):
    parser.add_argument('--save-dir', help='Directory for Triton CSV, PNG, and HTML reports')
    parser.add_argument('--plot', action='store_true', help='Also render faceted plots in --save-dir')


def run_module_benchmark(mark, save_dir: str | None = None, plot: bool = False) -> list[BenchmarkResult]:
    if plot and not save_dir:
        raise ValueError('--plot requires --save-dir')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    frames = mark.run(print_data=False, save_path=save_dir or '', return_df=True)
    configurations = mark.benchmarks if isinstance(mark.benchmarks, list) else [mark.benchmarks]
    frames = frames if isinstance(frames, list) else [frames]
    results = []
    for config, frame in zip(configurations, frames, strict=True):
        for values, row in zip(config.x_vals, frame.to_dict(orient='records'), strict=True):
            if not isinstance(values, list | tuple):
                values = [values for _ in config.x_names]
            parameters = {**dict(zip(config.x_names, values, strict=True)), **config.args}
            parameters = {key: str(value) if isinstance(value, torch.dtype) else value for key, value in parameters.items()}
            for provider in config.line_names:
                results.append(BenchmarkResult(
                    name=provider,
                    parameters=parameters,
                    metrics={'median_ms': float(row[provider])},
                ))
        if plot:
            from benchmarks.visualize import visualize_perf_report_csv
            for path in visualize_perf_report_csv(Path(save_dir) / f'{config.plot_name}.csv', save_dir=save_dir):
                print(f'Saved plot: {path}')
    return results
