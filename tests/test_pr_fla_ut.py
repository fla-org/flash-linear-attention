# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Hardware-independent regression tests for the PR gate runner.

The runner (.github/workflows/scripts/pr-fla-ut.py) decides whether the
PR gate passes, so its verdicts (pass/fail/timeout/skip), JUnit report
and CLI exit status are covered here without any accelerator.
"""

import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / ".github" / "workflows" / "scripts" / "pr-fla-ut.py"


def run_runner(tmp_path, commands, timeout=30, deadline=0):
    caselist = tmp_path / "casecmd.list"
    caselist.write_text("\n".join(commands) + "\n")
    xml_path = tmp_path / "results.xml"
    proc = subprocess.run(
        [
            sys.executable, str(RUNNER),
            "--caselist", str(caselist),
            "--build-dir", str(tmp_path),
            "--output-xml", str(xml_path),
            "--timeout", str(timeout),
            "--deadline", str(deadline),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc, xml_path


def parse_cases(xml_path):
    return ET.parse(xml_path).getroot().findall(".//testcase")


def test_all_pass_exit_zero(tmp_path):
    proc, xml_path = run_runner(tmp_path, ["echo hello", "true"])
    assert proc.returncode == 0
    cases = parse_cases(xml_path)
    assert len(cases) == 2
    assert all(c.find("failure") is None and c.find("skipped") is None for c in cases)


def test_nonzero_exit_marks_failure(tmp_path):
    proc, xml_path = run_runner(tmp_path, ["false"])
    assert proc.returncode == 1
    cases = parse_cases(xml_path)
    assert len(cases) == 1
    failure = cases[0].find("failure")
    assert failure is not None
    assert "Exit code: 1" in failure.get("message", "")


def test_timeout_kills_process_group(tmp_path):
    sleeper = (
        f'{sys.executable} -c "import os, time; '
        f"open('pid.txt', 'w').write(str(os.getpid())); time.sleep(30)\""
    )
    proc, xml_path = run_runner(tmp_path, [sleeper, "echo after"], timeout=1)
    assert proc.returncode == 1
    assert ">>> Result: [TIMEOUT]" in proc.stdout
    cases = parse_cases(xml_path)
    assert cases[0].find("failure") is not None
    assert "Timeout" in cases[0].find("failure").get("message", "")
    assert cases[1].find("failure") is None

    pid = int((tmp_path / "pid.txt").read_text())
    deadline = time.time() + 10
    dead = False
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            dead = True
            break
        time.sleep(0.2)
    assert dead, f"timed-out child process {pid} was not killed"


def test_invalid_utf8_does_not_abort_run(tmp_path):
    bad_utf8 = (
        f'{sys.executable} -c "import sys, time; '
        f"sys.stdout.buffer.write(b'\\xe2'); sys.stdout.buffer.flush(); time.sleep(5)\""
    )
    proc, xml_path = run_runner(tmp_path, [bad_utf8, "echo after"], timeout=1)
    assert "UnicodeDecodeError" not in proc.stderr
    assert proc.returncode == 1
    cases = parse_cases(xml_path)
    assert len(cases) == 2
    assert "Timeout" in cases[0].find("failure").get("message", "")
    assert cases[1].find("failure") is None


def test_deadline_skips_remaining_commands(tmp_path):
    proc, xml_path = run_runner(tmp_path, ["sleep 2", "echo late"], deadline=1)
    assert proc.returncode == 1
    cases = parse_cases(xml_path)
    assert cases[0].find("failure") is None
    skipped = cases[1].find("skipped")
    assert skipped is not None
    assert "deadline" in (skipped.get("message", "") or "")


def test_missing_caselist_exits_one(tmp_path):
    xml_path = tmp_path / "results.xml"
    proc = subprocess.run(
        [
            sys.executable, str(RUNNER),
            "--caselist", str(tmp_path / "absent.list"),
            "--build-dir", str(tmp_path),
            "--output-xml", str(xml_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1
    assert not xml_path.exists()
