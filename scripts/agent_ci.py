# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

PLATFORMS = {"h100", "ascend-a2"}


class AgentCIError(ValueError):
    pass


def load_catalog(path: Path) -> dict:
    catalog = json.loads(path.read_text())
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("suites"), dict):
        raise AgentCIError("Invalid agent test catalog")
    for name, suite in catalog["suites"].items():
        root = PurePosixPath(suite["root"])
        if root.is_absolute() or ".." in root.parts or not (suite["root"].startswith("tests/") or suite["root"] in {"tests", "scripts/tests"}):
            raise AgentCIError(f"Invalid root for suite {name}")
        if not suite.get("dispatch"):
            raise AgentCIError(f"Missing dispatch modes for suite {name}")
        platforms = suite.get("platforms", [])
        if not platforms or any(platform not in PLATFORMS for platform in platforms):
            raise AgentCIError(f"Invalid platforms for suite {name}")
    return catalog


def validate_platform(catalog: dict, suite_name: str, platform: str) -> None:
    if platform not in PLATFORMS:
        raise AgentCIError(f"Unsupported platform: {platform}")
    if platform not in catalog["suites"][suite_name]["platforms"]:
        raise AgentCIError(f"Suite {suite_name} is not supported on {platform}")


def validate_targets(catalog: dict, suite_name: str, raw_targets: str, repo: Path) -> list[str]:
    if suite_name not in catalog["suites"]:
        raise AgentCIError(f"Unknown test suite: {suite_name}")
    suite = catalog["suites"][suite_name]
    root = suite["root"]
    targets = raw_targets.split()
    if not targets:
        raise AgentCIError("At least one test target is required")
    for target in targets:
        file_part = target.split("::", 1)[0]
        path = PurePosixPath(file_part)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise AgentCIError(f"Invalid test target: {target}")
        if not file_part.startswith(f"{root}/"):
            raise AgentCIError(f"Test target is outside {root}: {target}")
        if suite.get("top_level_only") and len(path.parts) != 2:
            raise AgentCIError(f"Test target is not a top-level test: {target}")
        if not (repo / file_part).is_file():
            raise AgentCIError(f"Test target does not exist: {target}")
        if target.startswith("-"):
            raise AgentCIError(f"Pytest options are not test targets: {target}")
    return targets


def validate_dispatch(catalog: dict, suite_name: str, dispatch: str) -> list[str]:
    allowed = catalog["suites"][suite_name]["dispatch"]
    if dispatch not in allowed:
        raise AgentCIError(f"Dispatch mode {dispatch} is not allowed for {suite_name}")
    if dispatch == "both":
        return ["off", "on"]
    return [dispatch]


def read_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    result = {"tests": 0, "skipped": 0, "failures": 0, "errors": 0}
    for key in result:
        result[key] = sum(int(suite.attrib.get(key, "0")) for suite in suites)
    return result


def require_evidence(result: dict) -> None:
    if result["tests"] == 0:
        raise AgentCIError("Pytest collected zero tests; this is not passing evidence")
    if result["skipped"] == result["tests"]:
        raise AgentCIError("All tests were skipped; this is not passing evidence")


def run_targets(catalog: dict, repo: Path, suite: str, targets: list[str], dispatch: str, results: Path) -> dict:
    records = []
    for mode in validate_dispatch(catalog, suite, dispatch):
        for index, target in enumerate(targets):
            junit = results / f"{mode}-{index}.xml"
            env = os.environ.copy()
            if mode == "off":
                env["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
            else:
                env.pop("FLA_DISABLE_BACKEND_DISPATCH", None)
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-s", "-v", "--exitfirst", f"--junitxml={junit}", target],
                cwd=repo,
                env=env,
                check=False,
            )
            if not junit.is_file():
                raise AgentCIError(f"Pytest did not produce JUnit XML for {target} ({mode})")
            result = read_junit(junit)
            result.update({"target": target, "dispatch": mode, "exit_code": completed.returncode})
            records.append(result)
            require_evidence(result)
            if completed.returncode != 0:
                raise AgentCIError(f"Pytest failed for {target} ({mode}); see JUnit XML")
    return {"test_count": len(records), "results": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--platform", choices=sorted(PLATFORMS), required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--dispatch", choices=["off", "on", "both"], required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    catalog_path = args.catalog or repo / ".github/agent-tests/catalog.json"
    catalog = load_catalog(catalog_path)
    targets = validate_targets(catalog, args.suite, args.targets, repo)
    validate_platform(catalog, args.suite, args.platform)
    validate_dispatch(catalog, args.suite, args.dispatch)
    if args.validate_only:
        print(json.dumps({
            "valid": True,
            "platform": args.platform,
            "suite": args.suite,
            "targets": targets,
            "dispatch": args.dispatch,
        }))
        return 0
    if not args.results or not args.metadata:
        parser.error("--results and --metadata are required unless --validate-only is used")
    args.results.mkdir(parents=True, exist_ok=True)
    metadata = {
        "attempt_id": os.environ.get("FLA_AGENT_ATTEMPT_ID"),
        "base_sha": os.environ.get("FLA_AGENT_BASE_SHA"),
        "head_sha": os.environ.get("FLA_AGENT_HEAD_SHA"),
        "platform": args.platform,
        "suite": args.suite,
        "dispatch": args.dispatch,
    }
    try:
        summary = run_targets(catalog, repo, args.suite, targets, args.dispatch, args.results)
        metadata.update({"status": "passed", **summary})
    except AgentCIError as error:
        metadata.update({"status": "invalid_or_failed", "error": str(error)})
        args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AgentCIError as error:
        print(f"agent-ci error: {error}", file=sys.stderr)
        raise SystemExit(2)
