import ast
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def parse_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
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

        source_files = [p for s_dir in search_dirs for p in Path(s_dir).resolve().rglob("*.py") if p.name != '__init__.py']
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

    def find_dependent_tests(self, changed_files_str: list, max_depth=4) -> set:
        changed_files = {Path(f).resolve() for f in changed_files_str}

        # Heuristic: If a changed file is a modeling file, automatically include its config file at the start.
        initial_configs_to_add = set()
        for file in changed_files:
            if 'modeling_' in file.stem and 'models' in str(file):
                model_name = file.stem.replace('modeling_', '')
                config_file = file.parent / f"configuration_{model_name}.py"
                if config_file.is_file():
                    initial_configs_to_add.add(config_file)
        changed_files.update(initial_configs_to_add)

        all_affected_symbols = set()

        symbols_to_trace = set()
        for file_path in changed_files:
            if file_path.name == '__init__.py':
                continue
            new_defs = self.file_to_definitions.get(file_path, set())
            symbols_to_trace.update(new_defs)
            all_affected_symbols.update(new_defs)

        for i in range(max_depth):
            if not symbols_to_trace:
                break

            next_layer_files = set()
            for file_path, imported_symbols in self.file_to_imports.items():
                if not symbols_to_trace.isdisjoint(imported_symbols):
                    next_layer_files.add(file_path)

            # This heuristic is now also applied at each dependency level
            config_files_to_add = set()
            for file in next_layer_files:
                if 'modeling_' in file.stem and 'models' in str(file):
                    model_name = file.stem.replace('modeling_', '')
                    config_file = file.parent / f"configuration_{model_name}.py"
                    if config_file.is_file():
                        config_files_to_add.add(config_file)
            next_layer_files.update(config_files_to_add)

            new_definitions = set()
            for file_path in next_layer_files:
                new_definitions.update(self.file_to_definitions.get(file_path, set()))

            symbols_to_trace = new_definitions - all_affected_symbols
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
                dependent_tests.add(str(test_file))
                continue

            if not affected_source_file_stems.isdisjoint(imported_in_test):
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
