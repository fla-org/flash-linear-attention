# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import ast
import os
import sys
from collections import defaultdict
from functools import cache
from importlib.util import resolve_name
from pathlib import Path

DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() in ("true", "1", "yes")
DEBUG_TEST_FILE = os.environ.get("DEBUG_TEST_FILE", "NULL").lower()


@cache
def parse_file(file_path):
    try:
        with open(file_path, encoding="utf-8") as f:
            return ast.parse(f.read(), filename=file_path)
    except (SyntaxError, FileNotFoundError, UnicodeDecodeError):
        return None


def get_definitions_from_tree(tree) -> set:
    if not tree:
        return set()
    definitions = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            definitions.add(node.name)
    return definitions


def get_imports_from_tree(tree) -> set:
    """
    Return set of (module_path, symbol_name) tuples.
    module_path is the import source module (e.g., 'fla.ops.kda.wy_fast').
    symbol_name is the imported name (e.g., 'recompute_w_u_fwd').
    """
    if not tree:
        return set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.add((module, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # import x.y.z -> module="x.y", name="z"
                parts = alias.name.split('.')
                if len(parts) > 1:
                    module = '.'.join(parts[:-1])
                    name = parts[-1]
                else:
                    module = ""
                    name = parts[0]
                imports.add((module, name))
    return imports


def get_reexports_from_tree(tree, module: str, package: str) -> dict:
    """Resolve explicit symbol re-exports without importing implementation modules."""
    reexports = {}
    if not tree:
        return reexports
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            source = node.module or ''
            if node.level:
                source = resolve_name('.' * node.level + source, package)
            for alias in node.names:
                if alias.name != '*':
                    reexports[(module, alias.asname or alias.name)] = (source, alias.name)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    reexports[(module, target.id)] = (module, node.value.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            reexports.pop((module, node.name), None)
    return reexports


def _get_dispatch_owners(tree) -> set[str]:
    """Read operation keys from the supported dispatch imports."""
    if not tree:
        return set()
    dispatch_modules = {'fla.backends', 'fla.ops.backends', 'fla.modules.backends'}
    dispatch_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in dispatch_modules:
            dispatch_names.update(alias.asname or alias.name for alias in node.names if alias.name == 'dispatch')
    owners = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in dispatch_names:
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            continue
        operation = node.args[0].value
        if operation == 'modules.conv':
            owners.add('fla.modules.causal_conv1d.backends')
            continue
        if operation.startswith('modules.norm.'):
            owners.add('fla.modules.norm.triton_ascend')
            continue
        prefix = 'fla' if operation == 'modules' or operation.startswith('modules.') else 'fla.ops'
        owners.add(f'{prefix}.{operation}.backends')
    return owners


def find_backend_op_files(changed_files: list[str], project_root: Path) -> list[str]:
    """Map backend changes and shared kernel imports to dispatch consumers."""
    shared = any(file == 'fla/backends.py' or file.startswith('fla/ops/backends/') for file in changed_files)
    owners = set()
    for file in changed_files:
        parts = Path(file).parts
        if 'backends' in parts:
            owners.add('.'.join(parts[:parts.index('backends') + 1]))
        elif file in {
            'fla/modules/causal_conv1d/backends.py',
            'fla/modules/causal_conv1d/gluon.py',
            'fla/modules/causal_conv1d/triton_ascend.py',
        }:
            owners.add('fla.modules.causal_conv1d.backends')
        elif file.startswith('fla/modules/norm/triton_ascend/'):
            owners.add('fla.modules.norm.triton_ascend')
    if not shared and not owners:
        return []

    dispatch_owners = {}
    backend_dependents = defaultdict(set)
    for path in (project_root / 'fla').rglob('*.py'):
        relative_path = path.relative_to(project_root)
        parts = relative_path.parts
        tree = parse_file(path)
        if 'backends' in parts:
            owner = '.'.join(parts[:parts.index('backends') + 1])
            for module, _ in get_imports_from_tree(tree):
                module_parts = module.split('.')
                if 'backends' in module_parts:
                    dependency = '.'.join(module_parts[:module_parts.index('backends') + 1])
                    backend_dependents[dependency].add(owner)
        else:
            dispatch_owners[str(relative_path)] = _get_dispatch_owners(tree)

    pending = list(owners)
    while pending:
        for dependent in backend_dependents[pending.pop()] - owners:
            owners.add(dependent)
            pending.append(dependent)

    return [path for path, targets in dispatch_owners.items() if targets and (shared or owners & targets)]


def file_to_module_path(file_path: Path, project_root: Path) -> str:
    """Convert file path to Python module path."""
    try:
        rel_path = file_path.relative_to(project_root)
        # Remove .py extension and convert to module notation
        parts = list(rel_path.with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        return '.'.join(parts)
    except ValueError:
        return ""


class DependencyFinder:
    def __init__(self, search_dirs, test_dir, project_root=None):
        self.test_dir = Path(test_dir).resolve()
        self.project_root = Path(project_root).resolve() if project_root else self.test_dir.parent
        models_test_dir = self.test_dir / "models"

        all_source_files = [p for s_dir in search_dirs for p in Path(s_dir).resolve().rglob("*.py")]
        source_files = [p for p in all_source_files if p.name != '__init__.py']
        reexports = {}
        for file_path in all_source_files:
            module = file_to_module_path(file_path, self.project_root)
            package = module if file_path.name == '__init__.py' else module.rpartition('.')[0]
            reexports.update(get_reexports_from_tree(parse_file(file_path), module, package))
        test_scope = os.environ.get("TEST_SCOPE", "ALL").upper()
        if test_scope == "MODELS_ONLY":
            test_files = [p for p in models_test_dir.rglob("*.py") if p.name != '__init__.py']
        elif test_scope == "EXCLUDE_MODELS":
            all_files = self.test_dir.rglob("*.py")
            test_files = [p for p in all_files if p.name != '__init__.py' and models_test_dir not in p.parents]
        else:
            test_files = [p for p in self.test_dir.rglob("*.py") if p.name != '__init__.py']
        self.all_project_files = source_files + test_files
        self.all_test_files = set(test_files)

        # Build file path to module path mapping
        self.file_to_module = {}
        for file_path in self.all_project_files:
            mod = file_to_module_path(file_path, self.project_root)
            if mod:
                self.file_to_module[file_path] = mod

        self.file_to_definitions = {}
        self.file_to_imports = {}  # Now stores set of (module, symbol) tuples
        self.symbol_to_file_map = defaultdict(set)
        for file_path in self.all_project_files:
            tree = parse_file(file_path)
            definitions = get_definitions_from_tree(tree)
            imports = get_imports_from_tree(tree)
            for imported in list(imports):
                visited = set()
                while imported in reexports and imported not in visited:
                    visited.add(imported)
                    imported = reexports[imported]
                    imports.add(imported)
            self.file_to_definitions[file_path] = definitions
            self.file_to_imports[file_path] = imports
            for defn in definitions:
                self.symbol_to_file_map[defn].add(file_path)

    def _get_affected_modules(self, affected_files: set) -> set:
        """Get set of module paths for affected files."""
        affected_modules = set()
        for file_path in affected_files:
            mod = self.file_to_module.get(file_path)
            if mod:
                affected_modules.add(mod)
            # Also add parent modules for relative imports
            # e.g., fla.ops.gated_oja_rule.wy_fast -> fla.ops.gated_oja_rule
            if mod:
                parts = mod.split('.')
                for i in range(len(parts), 0, -1):
                    affected_modules.add('.'.join(parts[:i]))
        return affected_modules

    def _print_dependency_chain(self, symbol, module, symbol_chain_map):
        chain = []
        current_symbol = symbol
        while current_symbol is not None:
            file_path = next(iter(self.symbol_to_file_map.get(current_symbol, ["Unknown File"])), "Unknown File")
            chain.append(f"{current_symbol} @ {file_path}")
            current_symbol = symbol_chain_map.get(current_symbol)

        chain.reverse()
        print(f"  - Dependency Chain: {' -> '.join(chain)}", file=sys.stderr)

    def find_dependent_tests(self, changed_files_str: list, max_depth=4) -> set:
        changed_files = {Path(f).resolve() for f in changed_files_str}

        initial_configs_to_add = set()
        for file in changed_files:
            if 'modeling_' in file.stem and 'models' in str(file):
                model_name = file.stem.replace('modeling_', '')
                config_file = file.parent / f"configuration_{model_name}.py"
                if config_file.is_file():
                    initial_configs_to_add.add(config_file)
        changed_files.update(initial_configs_to_add)

        # Get affected module paths
        affected_modules = self._get_affected_modules(changed_files)

        symbol_chain_map = {}
        all_affected_symbols = set()
        symbols_to_trace = set()

        for file_path in changed_files:
            if file_path.name == '__init__.py':
                continue
            new_defs = self.file_to_definitions.get(file_path, set())
            symbols_to_trace.update(new_defs)
            all_affected_symbols.update(new_defs)
            for defn in new_defs:
                symbol_chain_map[defn] = None

        for i in range(max_depth):
            if not symbols_to_trace:
                break

            next_layer_files = set()
            newly_added_definitions = set()

            for file_path, imported in self.file_to_imports.items():
                # imported is a set of (module, symbol) tuples
                # Check if any import matches both:
                # 1. The symbol is in symbols_to_trace
                # 2. The module is in affected_modules (meaning it comes from an affected file)
                triggers = set()
                for mod, sym in imported:
                    if sym in symbols_to_trace and mod in affected_modules:
                        triggers.add(sym)

                if triggers:
                    next_layer_files.add(file_path)
                    defs_in_file = self.file_to_definitions.get(file_path, set())
                    first_trigger = next(iter(triggers))
                    for defn in defs_in_file:
                        if defn not in all_affected_symbols:
                            symbol_chain_map[defn] = first_trigger
                            newly_added_definitions.add(defn)

            # This heuristic is now also applied at each dependency level
            config_files_to_add = set()
            for file in next_layer_files:
                if 'modeling_' in file.stem and 'models' in str(file):
                    model_name = file.stem.replace('modeling_', '')
                    config_file = file.parent / f"configuration_{model_name}.py"
                    if config_file.is_file():
                        config_files_to_add.add(config_file)

            for config_file in config_files_to_add:
                defs_in_file = self.file_to_definitions.get(config_file, set())
                for defn in defs_in_file:
                    if defn not in all_affected_symbols:
                        symbol_chain_map[defn] = "CONFIG_HEURISTIC"
                        newly_added_definitions.add(defn)

            next_layer_files.update(config_files_to_add)
            # Update affected modules for next iteration
            affected_modules.update(self._get_affected_modules(next_layer_files))

            symbols_to_trace = newly_added_definitions
            all_affected_symbols.update(symbols_to_trace)

        dependent_tests = set()

        affected_source_file_stems = set()
        for s in all_affected_symbols:
            if s in self.symbol_to_file_map:
                for file_path in self.symbol_to_file_map[s]:
                    affected_source_file_stems.add(file_path.stem)

        for test_file in self.all_test_files:
            imported_in_test = self.file_to_imports.get(test_file, set())

            # Check if test imports any affected symbol from affected module
            test_imports_affected = False
            for mod, sym in imported_in_test:
                if sym in all_affected_symbols and mod in affected_modules:
                    test_imports_affected = True
                    if DEBUG_MODE and DEBUG_TEST_FILE in str(test_file).lower():
                        print(
                            f"DEBUG: Test file {test_file} is included because it imports "
                            f"{sym} from module {mod}", file=sys.stderr)
                        self._print_dependency_chain(sym, mod, symbol_chain_map)
                    break

            if test_imports_affected:
                dependent_tests.add(str(test_file))
                continue

            # Fallback: check if test imports a symbol matching affected file stem
            imported_symbols = {sym for _, sym in imported_in_test}
            if not affected_source_file_stems.isdisjoint(imported_symbols):
                if DEBUG_MODE and DEBUG_TEST_FILE in str(test_file).lower():
                    imported_files = [f for f in imported_symbols if f in affected_source_file_stems]
                    print(
                        f"DEBUG: Test file {test_file} is included because it imports "
                        f"a symbol matching an affected file stem: {imported_files}", file=sys.stderr)
                dependent_tests.add(str(test_file))

        for changed_file in changed_files:
            if changed_file in self.all_test_files:
                dependent_tests.add(str(changed_file))

        # Filter out test files containing 'mamba'
        dependent_tests = {test for test in dependent_tests if 'mamba' not in os.path.basename(test)}

        return dependent_tests


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_dependent_tests.py <file1> <file2> ...")
        sys.exit(1)

    all_args_string = " ".join(sys.argv[1:])
    changed_files = all_args_string.split()

    BLACKLIST = ['fla/utils.py', 'utils/convert_from_llama.py', 'utils/convert_from_rwkv6.py',
                 'utils/convert_from_rwkv7.py', 'tests/conftest.py']
    # `fla/utils.py` was split into `fla/utils/**`; keep the same test-selection behavior
    # for broad utility changes instead of accidentally unblacklisting the new package paths.
    BLACKLIST_PREFIXES = ['fla/utils/']
    changed_files = [
        file for file in changed_files
        if not any(file.endswith(b) for b in BLACKLIST)
        and not any(file.startswith(b) for b in BLACKLIST_PREFIXES)
    ]

    changed_files = [file for file in changed_files if file.endswith('.py')]

    current_dir = Path(__file__).parent.resolve()
    test_dir = current_dir.parent / "tests"
    search_dir = current_dir.parent / "fla"
    project_root = current_dir.parent

    additional_files = find_backend_op_files(changed_files, project_root)
    if additional_files:
        changed_files = list(dict.fromkeys(changed_files + additional_files))

    finder = DependencyFinder(search_dirs=[search_dir], test_dir=test_dir, project_root=project_root)
    dependent_tests = finder.find_dependent_tests(changed_files)

    if dependent_tests:
        print(" ".join(sorted(list(dependent_tests))))
