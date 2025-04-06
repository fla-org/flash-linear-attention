import ast
import sys
from pathlib import Path


def extract_definitions(file_path):
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


def find_dependent_tests(changed_file, test_dir, search_dir):
    """Find test files that call any of the definitions in the changed file or files that use these definitions."""
    # Convert all paths to absolute paths for consistent comparison
    abs_test_dir = Path(test_dir).resolve()
    abs_search_dir = Path(search_dir).resolve()
    abs_changed_file = Path(changed_file).resolve()

    # Check if the changed file is a test file itself
    is_test_file = abs_test_dir in abs_changed_file.parents

    # Extract definitions from the changed file
    definitions = extract_definitions(abs_changed_file)
    if not definitions and not is_test_file:
        return set()

    # If it's a test file, just return itself
    if is_test_file:
        return {str(abs_changed_file)}

    # Find files that use these definitions
    dependent_files = find_files_using_definitions(definitions, abs_search_dir)

    # Find test files that call these definitions or are dependent on the dependent files
    test_files = set()
    for test_file in abs_test_dir.rglob("test_*.py"):
        if find_calls_in_file(test_file, definitions):
            test_files.add(str(test_file))
        else:
            for dependent_file in dependent_files:
                if find_calls_in_file(test_file, extract_definitions(dependent_file)):
                    test_files.add(str(test_file))
                    break
    return test_files


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
        'fla/ops/utils/testing.py',
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
