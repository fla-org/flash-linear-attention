# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def run_target(target: str) -> tuple[str, int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-s", "-v", "--exitfirst", target],
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + (f"\n{completed.stderr}" if completed.stderr else "")
    return target, completed.returncode, output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run independent pytest files with bounded parallelism")
    parser.add_argument("targets", nargs="+", help="Pytest files to run in separate processes")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("FLA_TEST_JOBS", "2")))
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(run_target, args.targets))
    exit_code = 0
    for target, returncode, output in results:
        print(f"\n===== {target} =====\n{output}", flush=True)
        if exit_code == 0 and returncode != 0:
            exit_code = returncode
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
