# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run the current benchmark suite against isolated source revisions."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_project_root() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()


def get_commit_sha(ref: str) -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', '--verify', f'{ref}^{{commit}}'],
        cwd=find_project_root(),
        text=True,
    ).strip()


def run_benchmark(module: str, arguments: list[str], ref: str | None = None) -> tuple[list[dict], dict]:
    """Run one worker with the current benchmark definitions and the selected FLA revision.

    A missing ref uses the working tree, including uncommitted edits. Each worker exits before the next run,
    releasing accelerator memory. Named refs use detached worktrees without changing the installed package.
    """
    root = find_project_root()
    sha = get_commit_sha(ref=ref) if ref is not None else None
    with tempfile.TemporaryDirectory(prefix='fla_benchmark_') as temporary:
        temporary = Path(temporary)
        harness = temporary / 'harness'
        shutil.copytree(
            src=Path(__file__).resolve().parents[1],
            dst=harness / 'benchmarks',
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
        )
        worktree = temporary / 'worktree'
        try:
            if sha is not None:
                subprocess.run(
                    ['git', 'worktree', 'add', '--detach', str(worktree), sha],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            source = str(worktree) if sha is not None else root
            output = temporary / 'results.json'
            # the snapshot supplies benchmarks; the selected source tree supplies fla.
            bootstrap = (
                'import runpy, sys; '
                'sys.path[:0] = sys.argv[1:3]; '
                'module = sys.argv[3]; '
                'sys.argv = sys.argv[3:]; '
                'runpy.run_module(module, run_name="__main__")'
            )
            command = [sys.executable, '-c', bootstrap, str(harness), source, module, *arguments, '--json', str(output)]
            environment = {**os.environ, 'FLA_BENCH_WORKER': '1', 'FLA_GRAPH_BENCH_WORKER': '1'}
            subprocess.run(command, cwd=source, env=environment, check=True)
            with output.open() as file:
                payload = json.load(file)
            if ref is not None:
                payload['machine_info']['git_label'] = f'{ref}[{sha[:7]}]'
            return payload['results'], payload['machine_info']
        finally:
            if worktree.exists():
                subprocess.run(
                    ['git', 'worktree', 'remove', '--force', str(worktree)],
                    cwd=root,
                    capture_output=True,
                    check=True,
                )
