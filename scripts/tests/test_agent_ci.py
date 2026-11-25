# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("agent_ci", Path(__file__).parents[1] / "agent_ci.py")
agent_ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent_ci)


class AgentCITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        (self.repo / "tests/ops").mkdir(parents=True)
        (self.repo / "tests/ops/test_ok.py").write_text("")
        self.catalog = {
            "schema_version": 1,
            "suites": {
                "ops": {
                    "root": "tests/ops",
                    "platforms": ["h100", "ascend-a2", "ascend-a5"],
                    "dispatch": ["off", "on", "both"],
                },
            },
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_accepts_target_inside_suite(self):
        targets = agent_ci.validate_targets(self.catalog, "ops", "tests/ops/test_ok.py::test_ok", self.repo)
        self.assertEqual(targets, ["tests/ops/test_ok.py::test_ok"])

    def test_rejects_path_outside_suite(self):
        with self.assertRaises(agent_ci.AgentCIError):
            agent_ci.validate_targets(self.catalog, "ops", "tests/models/test_ok.py", self.repo)

    def test_rejects_parent_traversal(self):
        with self.assertRaises(agent_ci.AgentCIError):
            agent_ci.validate_targets(self.catalog, "ops", "tests/ops/../models/test_ok.py", self.repo)

    def test_rejects_missing_target(self):
        with self.assertRaises(agent_ci.AgentCIError):
            agent_ci.validate_targets(self.catalog, "ops", "tests/ops/missing.py", self.repo)

    def test_accepts_suite_root_directory(self):
        utils_catalog = {
            "suites": {
                "utils": {"root": "tests/utils", "platforms": ["ascend-a5"], "dispatch": ["on"]},
            },
        }
        (self.repo / "tests/utils").mkdir(parents=True, exist_ok=True)
        (self.repo / "tests/utils/test_ok.py").write_text("")
        targets = agent_ci.validate_targets(utils_catalog, "utils", "tests/utils", self.repo)
        self.assertEqual(targets, ["tests/utils"])

    def test_both_dispatch_modes(self):
        self.assertEqual(agent_ci.validate_dispatch(self.catalog, "ops", "both"), ["on", "off"])

    def test_failing_leg_does_not_skip_remaining_targets_or_modes(self):
        from unittest import mock

        def fake_pytest(command, cwd, env, check):
            target = command[-1]
            junit = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
            failing = target.endswith("test_bad") and env.get("FLA_DISABLE_BACKEND_DISPATCH") == "1"
            failures = 1 if failing else 0
            junit.write_text(f'<testsuite tests="2" skipped="0" failures="{failures}" errors="0"/>')
            return mock.Mock(returncode=1 if failing else 0)

        results = self.repo / "junit"
        results.mkdir()
        with mock.patch.object(agent_ci.subprocess, "run", side_effect=fake_pytest) as run:
            summary = agent_ci.run_targets(
                self.catalog,
                self.repo,
                "ascend-a2",
                "ops",
                ["tests/ops/test_ok.py::test_bad", "tests/ops/test_ok.py::test_ok"],
                "both",
                results,
            )

        modes = [call.kwargs["env"].get("FLA_DISABLE_BACKEND_DISPATCH") for call in run.call_args_list]
        self.assertEqual(modes, [None, None, "1", "1"])
        self.assertEqual(summary["test_count"], 4)
        self.assertEqual(len(summary["failures"]), 1)
        self.assertIn("tests/ops/test_ok.py::test_bad (off)", summary["failures"][0])
        self.assertTrue((results / "on-0.xml").is_file())
        self.assertTrue((results / "on-1.xml").is_file())
        self.assertTrue((results / "off-0.xml").is_file())
        self.assertTrue((results / "off-1.xml").is_file())

    def test_top_level_suite_rejects_nested_tests(self):
        catalog = {"suites": {"top-level": {"root": "tests",
                                            "top_level_only": True, "platforms": ["h100"], "dispatch": ["off"]}}}
        (self.repo / "tests/test_root.py").write_text("")
        targets = agent_ci.validate_targets(catalog, "top-level", "tests/test_root.py", self.repo)
        self.assertEqual(targets, ["tests/test_root.py"])
        with self.assertRaises(agent_ci.AgentCIError):
            agent_ci.validate_targets(catalog, "top-level", "tests/ops/test_ok.py", self.repo)

    def test_platform_must_support_suite(self):
        agent_ci.validate_platform(self.catalog, "ops", "h100")
        agent_ci.validate_platform(self.catalog, "ops", "ascend-a2")
        agent_ci.validate_platform(self.catalog, "ops", "ascend-a5")
        with self.assertRaises(agent_ci.AgentCIError):
            agent_ci.validate_platform(self.catalog, "ops", "amd-mi300")

    def test_repository_catalog_supports_targeted_ascend_a5_suites(self):
        catalog = agent_ci.load_catalog(Path(__file__).parents[2] / ".github/agent-tests/catalog.json")
        for suite in ("ops", "modules", "cp", "top-level", "op-utils", "utils", "benchmarks"):
            agent_ci.validate_platform(catalog, suite, "ascend-a5")
        for suite in ("scripts", "layers", "models"):
            with self.assertRaises(agent_ci.AgentCIError):
                agent_ci.validate_platform(catalog, suite, "ascend-a5")

    def test_zero_tests_is_not_evidence(self):
        with self.assertRaises(agent_ci.AgentCIError):
            agent_ci.require_evidence({"tests": 0, "skipped": 0})

    def test_all_skipped_is_not_evidence(self):
        with self.assertRaises(agent_ci.AgentCIError):
            agent_ci.require_evidence({"tests": 2, "skipped": 2})

    def test_junit_counts_results(self):
        path = self.repo / "result.xml"
        path.write_text('<testsuite tests="3" skipped="1" failures="1" errors="0"/>')
        result = agent_ci.read_junit(path)
        self.assertEqual(result, {"tests": 3, "skipped": 1, "failures": 1, "errors": 0})

    def test_catalog_rejects_invalid_root(self):
        path = self.repo / "catalog.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "suites": {"bad": {"root": "../ops", "platforms": ["h100"], "dispatch": ["off"]}},
        }))
        with self.assertRaises(agent_ci.AgentCIError):
            agent_ci.load_catalog(path)

    def test_catalog_requires_supported_platform(self):
        path = self.repo / "catalog.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "suites": {"ops": {"root": "tests/ops", "platforms": ["amd-mi300"], "dispatch": ["off"]}},
        }))
        with self.assertRaises(agent_ci.AgentCIError):
            agent_ci.load_catalog(path)


if __name__ == "__main__":
    unittest.main()
