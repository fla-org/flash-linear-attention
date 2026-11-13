# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def write_test(path: Path, body: str) -> None:
    path.write_text(f"import time\n\ndef test_example():\n    time.sleep(0.05)\n    {body}\n")


def run_helper(*args: str) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_tests_parallel.py"
    return subprocess.run([sys.executable, str(script), *args], text=True, capture_output=True, check=False)


def test_parallel_runner_executes_real_pytest_files(tmp_path: Path):
    first = tmp_path / "test_first.py"
    second = tmp_path / "test_second.py"
    write_test(first, "assert True")
    write_test(second, "assert True")
    result = run_helper("--jobs", "2", str(first), str(second))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("1 passed") == 2
    assert result.stdout.index("test_first.py") < result.stdout.index("test_second.py")


def test_parallel_runner_reports_all_results_on_failure(tmp_path: Path):
    first = tmp_path / "test_first.py"
    second = tmp_path / "test_second.py"
    write_test(first, "assert False")
    write_test(second, "assert True")
    result = run_helper("--jobs", "2", str(first), str(second))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "assert False" in result.stdout
    assert "1 failed" in result.stdout
    assert "1 passed" in result.stdout


def test_parallel_runner_rejects_unbounded_jobs(tmp_path: Path):
    target = tmp_path / "test_example.py"
    write_test(target, "assert True")
    result = run_helper("--jobs", "5", str(target))
    assert result.returncode == 2
    assert "between 1 and 4" in result.stderr
