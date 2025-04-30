import ast
import sys
from collections import deque
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=200)
def extract_definitions(file_path, include_vars=False):
    """Extract function and class definitions from a Python file."""
    if file_path.suffix != ".py":
        return set()
    try:
        with open(file_path, "r") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                return set()
        definitions = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions.add(node.name)
            elif include_vars and isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        definitions.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                definitions.add(elt.id)
        return definitions
    except BaseException:
        return set()


def find_calls_in_file(file_path, definitions):
    """Check if a file calls any of the given definitions."""
    if file_path.suffix != ".py":
        return False
    with open(file_path, "r") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in definitions:
                return True
    return False


def find_files_using_definitions(definitions, directory):
    """Find files in the directory that use any of the given definitions."""
    files = []
    for file_path in Path(directory).rglob("*.py"):
        if find_calls_in_file(file_path, definitions):
            files.append(file_path)
    return files


def find_dependent_tests(changed_file, test_dir, search_dir, max_depth=3):
    abs_test_dir = Path(test_dir).resolve()
    abs_search_dir = Path(search_dir).resolve()
    abs_changed_file = Path(changed_file).resolve()

    if abs_test_dir in abs_changed_file.parents:
        return {str(abs_changed_file)}

    queue = deque([(abs_changed_file, 0, True)])  # (file, depth, include_vars)
    processed = set()
    all_definitions = set()

    while queue:
        current_file, depth, include_vars = queue.popleft()
        if current_file in processed or depth > max_depth:
            continue

        processed.add(current_file)
        current_defs = extract_definitions(current_file, include_vars)
        all_definitions.update(current_defs)

        for f in Path(abs_search_dir).rglob("*.py"):
            if f not in processed and find_calls_in_file(f, current_defs):
                queue.append((f, depth + 1, False))

    return {
        str(test_file) for test_file in abs_test_dir.rglob("test_*.py")
        if find_calls_in_file(test_file, all_definitions)
    }


if __name__ == "__main__":
    # Get the changed files from the command line arguments
    if len(sys.argv) < 2:
        print("Usage: python scripts/find_dependent_tests.py <file_paths>")
        sys.exit(1)

    # Split the input argument into individual file paths
    changed_files = [Path(file) for file in sys.argv[1].split()]
    # Skip fla/utils.py
    BLACKLIST = [
        'fla/utils.py',
        'utils/convert_from_llama.py',
        'utils/convert_from_rwkv6.py',
        'utils/convert_from_rwkv7.py',
    ]
    changed_files = [
        file for file in changed_files
        if not any(str(file).endswith(blacklisted) for blacklisted in BLACKLIST)
    ]

    # Define the test directory and the directory to search for dependent files
    current_dir = Path(__file__).parent.resolve()
    test_dir = current_dir.parent / "tests"
    search_dir = current_dir.parent / "fla"

    # Find dependent test files for each changed file
    dependent_tests = set()
    for changed_file in changed_files:
        dependent_tests.update(find_dependent_tests(changed_file, test_dir, search_dir))

    # Output the test files as a space-separated string
    print(" ".join(dependent_tests))
