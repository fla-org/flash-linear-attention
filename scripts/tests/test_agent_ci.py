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
                    "platforms": ["h100", "ascend-a2"],
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

    def test_both_dispatch_modes(self):
        self.assertEqual(agent_ci.validate_dispatch(self.catalog, "ops", "both"), ["off", "on"])

    def test_platform_must_support_suite(self):
        agent_ci.validate_platform(self.catalog, "ops", "h100")
        agent_ci.validate_platform(self.catalog, "ops", "ascend-a2")
        with self.assertRaises(agent_ci.AgentCIError):
            agent_ci.validate_platform(self.catalog, "ops", "amd-mi300")

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
