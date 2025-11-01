import ast
import os
import sys
from collections import defaultdict
from functools import cache
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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.add(node.name)
    return definitions


def get_imports_from_tree(tree) -> set:
    if not tree:
        return set()
    imports = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.asname or alias.name.split('.')[0])
    return imports


class DependencyFinder:
    def __init__(self, search_dirs, test_dir):
        self.test_dir = Path(test_dir).resolve()
        models_test_dir = self.test_dir / "models"

        source_files = [p for s_dir in search_dirs for p in Path(s_dir).resolve().rglob("*.py") if p.name != '__init__.py']
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

        self.file_to_definitions = {}
        self.file_to_imports = {}
        self.symbol_to_file_map = defaultdict(set)
        for file_path in self.all_project_files:
            tree = parse_file(file_path)
            definitions = get_definitions_from_tree(tree)
            imports = get_imports_from_tree(tree)
            self.file_to_definitions[file_path] = definitions
            self.file_to_imports[file_path] = imports
            for defn in definitions:
                self.symbol_to_file_map[defn].add(file_path)

    def _print_dependency_chain(self, symbol, symbol_chain_map):
        chain = []
        current_symbol = symbol
        while current_symbol is not None:
            # Find the file where the symbol is defined
            # In case of multiple definitions, we take the first one found
            file_path = next(iter(self.symbol_to_file_map.get(current_symbol, ["Unknown File"])), "Unknown File")
            chain.append(f"{current_symbol} @ {file_path}")
            current_symbol = symbol_chain_map.get(current_symbol)

        chain.reverse()
        print("  - Dependency Chain:", " -> ".join(chain), file=sys.stderr)

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

            # Find files that import the current symbols to trace
            # And for each new definition, link it to the symbol that triggered it
            newly_added_definitions = set()
            for file_path, imported_symbols in self.file_to_imports.items():
                triggers = symbols_to_trace.intersection(imported_symbols)
                if triggers:
                    next_layer_files.add(file_path)
                    defs_in_file = self.file_to_definitions.get(file_path, set())
                    # For simplicity, we link all new definitions in this file to the first trigger found
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

            # For config files, we don't have a clear trigger, so we can't map their chain
            for config_file in config_files_to_add:
                defs_in_file = self.file_to_definitions.get(config_file, set())
                for defn in defs_in_file:
                    if defn not in all_affected_symbols:
                        symbol_chain_map[defn] = "CONFIG_HEURISTIC"  # Special marker
                        newly_added_definitions.add(defn)

            next_layer_files.update(config_files_to_add)

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

            if not all_affected_symbols.isdisjoint(imported_in_test):
                if DEBUG_MODE and DEBUG_TEST_FILE in str(test_file).lower():
                    imported_symbols = [s for s in imported_in_test if s in all_affected_symbols]
                    print(
                        f"DEBUG: Test file {test_file} is included because it imports affected symbols: {imported_symbols}", file=sys.stderr)  # noqa: E501
                    for symbol in imported_symbols:
                        self._print_dependency_chain(symbol, symbol_chain_map)
                dependent_tests.add(str(test_file))
                continue

            if not affected_source_file_stems.isdisjoint(imported_in_test):
                if DEBUG_MODE and DEBUG_TEST_FILE in str(test_file).lower():
                    imported_files = [f for f in imported_in_test if f in affected_source_file_stems]
                    print(
                        f"DEBUG: Test file {test_file} is included because it imports a symbol matching an affected file stem: {imported_files}", file=sys.stderr)  # noqa: E501
                dependent_tests.add(str(test_file))
        for changed_file in changed_files:
            if changed_file in self.all_test_files:
                dependent_tests.add(str(changed_file))

        return dependent_tests


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_dependent_tests.py <file1> <file2> ...")
        sys.exit(1)

    all_args_string = " ".join(sys.argv[1:])
    changed_files = all_args_string.split()

    BLACKLIST = ['fla/utils.py', 'utils/convert_from_llama.py', 'utils/convert_from_rwkv6.py', 'utils/convert_from_rwkv7.py']
    changed_files = [file for file in changed_files if not any(file.endswith(b) for b in BLACKLIST)]

    changed_files = [file for file in changed_files if file.endswith('.py')]

    current_dir = Path(__file__).parent.resolve()
    test_dir = current_dir.parent / "tests"
    search_dir = current_dir.parent / "fla"

    finder = DependencyFinder(search_dirs=[search_dir], test_dir=test_dir)
    dependent_tests = finder.find_dependent_tests(changed_files)

    if dependent_tests:
        print(" ".join(sorted(list(dependent_tests))))
