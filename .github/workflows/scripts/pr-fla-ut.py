# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run FLA PR tests from a caselist, generating a JUnit XML report.

Adapted from FlashMLA-for-sail's pr-flashmla-ut.py.  Each non-empty,
non-comment line in the caselist is executed as a shell command in a fresh
subprocess. The runner records pytest's exit status and output, then writes
test-results.xml. If any case fails or times out, the process exits with code 1.
"""

import argparse
import contextlib
import os
import re
import signal
import subprocess
import sys
import time

from junit_xml import TestCase, TestSuite

OUTPUT_XML = "test-results.xml"


def derive_test_name(case_command: str) -> str:
    return re.sub(r"\s+", "_", case_command.strip())


def run_one_case(i: int, case_command: str, total_count: int, timeout: float = 1200):
    """Run one caselist command; return (junit_case, status, reason, output).

    status is one of PASS/FAIL/TIMEOUT/ERROR and is always printed as
    '>>> Result: [<status>]' so log search can find every verdict.
    """
    test_name = derive_test_name(case_command)
    junit_case = TestCase(name=test_name, classname="flaTest")

    print(f"\n[{i}/{total_count}] Running: {test_name}")
    print(f"Command: {case_command}")
    print("-" * 50)

    start_time = time.time()
    stdout_output: list[str] = []

    def emit(output: str) -> str:
        if output:
            print(output, end="")
            stdout_output.append(output)
        return "".join(stdout_output)

    try:
        process = subprocess.Popen(
            case_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )

        output, _ = process.communicate(timeout=timeout)
        collected = emit(output)
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration

        junit_case.stdout = output
        # Pytest uses nonzero statuses for failures, collection errors and no tests.
        if process.returncode == 0:
            print(">>> Result: [PASS]")
            return junit_case, "PASS", "", collected

        reason = f"Exit code: {process.returncode}"
        print(f">>> Result: [FAIL] ({reason})")
        junit_case.add_failure_info(message=reason, output=collected)
        return junit_case, "FAIL", reason, collected

    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        collected = emit(output)
        junit_case.elapsed_sec = time.time() - start_time
        reason = f"Timeout after {timeout:g}s"
        print(f">>> Result: [TIMEOUT] ({reason})")
        junit_case.add_failure_info(message=reason, output=collected or "Test timed out")
        return junit_case, "TIMEOUT", reason, collected

    except Exception as e:
        junit_case.elapsed_sec = time.time() - start_time
        reason = f"Exception: {e}"
        print(f">>> Result: [ERROR] ({reason})")
        junit_case.add_error_info(message=reason, output=str(e))
        return junit_case, "ERROR", reason, "".join(stdout_output)


def parse_commands(filepath: str) -> list[str]:
    commands: list[str] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            commands.append(line)
    return commands


def write_xml(xml_path: str, cases: list[TestCase]) -> None:
    """Persist the JUnit report so partial results survive external kills."""
    os.makedirs(os.path.dirname(xml_path) or ".", exist_ok=True)
    with open(xml_path, "w", encoding="utf-8") as f:
        TestSuite.to_file(f, [TestSuite("FLA PR Tests", cases)], prettyprint=True)


def run_cases(caselist_path: str, build_dir: str, output_xml: str = OUTPUT_XML, timeout: float = 1200, deadline: float = 0):
    if not os.path.exists(caselist_path):
        print(f"Error: caselist not found: {caselist_path}")
        sys.exit(1)

    test_commands = parse_commands(caselist_path)
    if not test_commands:
        print("No valid test commands found")
        sys.exit(1)

    os.chdir(build_dir)
    print(f"Working directory: {os.getcwd()}")

    xml_path = os.path.abspath(output_xml)
    all_cases: list[TestCase] = []
    failures: list[tuple[str, str, str, str]] = []
    passed = 0
    start_time = time.time()

    for i, cmd in enumerate(test_commands, 1):
        if deadline and time.time() - start_time > deadline:
            junit_case = TestCase(name=derive_test_name(cmd), classname="flaTest")
            reason = "Not run: overall deadline exceeded"
            junit_case.add_skipped_info(reason)
            all_cases.append(junit_case)
            failures.append(("SKIPPED", derive_test_name(cmd), reason, ""))
            write_xml(xml_path, all_cases)
            continue
        junit_case, status, reason, output = run_one_case(i, cmd, len(test_commands), timeout)
        all_cases.append(junit_case)
        if status == "PASS":
            passed += 1
        else:
            failures.append((status, derive_test_name(cmd), reason, output))
        write_xml(xml_path, all_cases)

    total = len(test_commands)
    print(f"\nTests completed. Total: {total}, Passed: {passed}")
    print(f"Report: {xml_path}")

    if failures:
        # Echo a diagnosis block at the end of the log so the failing case and
        # its last output lines are visible without scrolling the whole run.
        print("\n=== FAILED CASES ===")
        for status, name, reason, output in failures:
            print(f"[{status}] {name}")
            print(f"    Reason: {reason}")
            tail = output.splitlines()[-40:]
            if tail:
                print("    Last output lines:")
                for line in tail:
                    print(f"      {line}")
        print("====================")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLA PR tests from a caselist.")
    parser.add_argument("--caselist", required=True, help="Path to the caselist file")
    parser.add_argument(
        "--build-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        help="Working directory for test commands (default: repo root)",
    )
    parser.add_argument("--output-xml", default=OUTPUT_XML, help="JUnit XML output path")
    parser.add_argument("--timeout", type=float, default=1200, help="Per-command timeout in seconds")
    parser.add_argument(
        "--deadline",
        type=float,
        default=0,
        help="Overall test budget in seconds; later commands are skipped (0 disables)",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.deadline < 0:
        parser.error("--deadline must not be negative")
    run_cases(os.path.abspath(args.caselist), os.path.abspath(args.build_dir), args.output_xml, args.timeout, args.deadline)
