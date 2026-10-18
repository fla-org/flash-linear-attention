# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Select operator benchmarks affected by a source diff."""

import ast
import re
import subprocess
from pathlib import Path

from benchmarks.ops.utils.registry import get_op, list_ops
from benchmarks.utils.revisions import find_project_root

_OP_DIR_RE = re.compile(r'^fla/ops/([^/]+)/')
_SHARED_BENCHMARK_FILES = {
    'benchmarks/compare.py',
    'benchmarks/utils/revisions.py',
    'benchmarks/ops/utils/discovery.py',
    'benchmarks/utils/environment.py',
    'benchmarks/ops/utils/inputs.py',
    'benchmarks/ops/utils/registry.py',
    'benchmarks/ops/benchmark_ops.py',
    'benchmarks/ops/utils/runner.py',
}


def get_changed_files(base: str, head: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", base, head, "--", "*.py"],
        cwd=find_project_root(),
        text=True,
    ).strip()
    return [f for f in out.split('\n') if f]


def _imported_op_dirs(relpath: str) -> set[str]:
    path = Path(find_project_root()) / relpath
    if not path.is_file():
        return set()
    try:
        text = path.read_text(encoding='utf-8')
        tree = ast.parse(text, filename=relpath)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()

    subdirs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if mod == 'fla.ops' or mod.startswith('fla.ops.'):
                parts = mod.split('.')
                if len(parts) >= 3 and parts[0] == 'fla' and parts[1] == 'ops':
                    subdirs.add(parts[2])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith('fla.ops.'):
                    parts = name.split('.')
                    if len(parts) >= 3 and parts[0] == 'fla' and parts[1] == 'ops':
                        subdirs.add(parts[2])
    return subdirs


def find_affected_op_names(changed_files: list[str]) -> list[str]:
    import_map: dict[str, list[str]] = {}
    for name in list_ops():
        config = get_op(name)
        parts = config.import_path.split('.')
        if len(parts) >= 3 and parts[0] == 'fla' and parts[1] == 'ops':
            import_map.setdefault(parts[2], []).append(name)

    affected_dirs: set[str] = set()
    common_changed = False
    for fpath in changed_files:
        if not fpath.endswith('.py'):
            continue
        if fpath in _SHARED_BENCHMARK_FILES or fpath.startswith(('fla/ops/common/', 'fla/ops/utils/')):
            common_changed = True
            continue
        match = _OP_DIR_RE.match(fpath)
        if match:
            affected_dirs.add(match.group(1))

        if fpath.startswith('fla/') and not match:
            affected_dirs.update(_imported_op_dirs(relpath=fpath))

    if common_changed:
        return list_ops()

    op_names: set[str] = set()
    for directory in affected_dirs:
        if directory in import_map:
            op_names.update(import_map[directory])
    return sorted(op_names)
