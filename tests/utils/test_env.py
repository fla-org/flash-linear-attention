# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib.util
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location('fla_env', Path(__file__).parents[2] / 'fla/utils/env.py')
env = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(env)


class TritonVersionWarningTest(unittest.TestCase):
    def setUp(self):
        self.messages = []

        class ListHandler(logging.Handler):
            def emit(handler_self, record):
                self.messages.append(record.getMessage())

        self.handler = ListHandler(level=logging.WARNING)
        env.logger.addHandler(self.handler)

    def tearDown(self):
        env.logger.removeHandler(self.handler)
        env.check_environments.cache_clear()

    def run_check(self, version, backend='cuda', arch=90):
        self.messages.clear()
        target = SimpleNamespace(backend=backend, arch=arch)
        driver = SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: target))
        env.check_environments.cache_clear()
        with (
            mock.patch.object(env.triton, '__version__', version),
            mock.patch.object(env.triton.runtime, 'driver', driver),
        ):
            env.check_environments()
        return self.messages

    def test_blackwell_below_3_7_1_warns_illegal_memory_access(self):
        for arch in (100, '100f', 120):
            messages = self.run_check('3.3.1', arch=arch)
            self.assertEqual(len(messages), 1)
            self.assertIn('Blackwell', messages[0])
            self.assertIn('>= 3.7.1', messages[0])
            self.assertIn('3.9.0 recommended', messages[0])
            self.assertIn('illegal memory accesses', messages[0])

    def test_blackwell_below_3_9_0_recommends_upgrade(self):
        messages = self.run_check('3.7.1', arch=100)
        self.assertEqual(len(messages), 1)
        self.assertIn('Blackwell', messages[0])
        self.assertIn('3.9.0 or newer is recommended', messages[0])
        self.assertNotIn('illegal memory accesses', messages[0])

    def test_hopper_below_3_9_0_recommends_upgrade(self):
        messages = self.run_check('3.5.1', arch=90)
        self.assertEqual(len(messages), 1)
        self.assertIn('Hopper', messages[0])
        self.assertIn('3.9.0 or newer is recommended', messages[0])

    def test_3_9_0_does_not_warn_on_hopper_or_blackwell(self):
        for arch in (90, 100, 120):
            self.assertEqual(self.run_check('3.9.0', arch=arch), [])

    def test_non_nvidia_does_not_get_nvidia_warning(self):
        self.assertEqual(self.run_check('3.5.0', backend='npu', arch=''), [])

    def test_below_3_3_keeps_existing_floor_warning(self):
        messages = self.run_check('3.2.0', arch=90)
        self.assertEqual(len(messages), 1)
        self.assertIn('below the recommended 3.3.0', messages[0])


if __name__ == '__main__':
    unittest.main()
