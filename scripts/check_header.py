# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Check and fix copyright headers in Python source files.

Usage:
    python scripts/check_header.py --check              # check only, exit 1 on failure
    python scripts/check_header.py --fix                # fix all files in-place
    python scripts/check_header.py --fix --year 2027    # fix and update the end year
"""

import argparse
import datetime
import sys
from pathlib import Path

REPO_URL = "https://github.com/fla-org/flash-linear-attention"
START_YEAR = 2023
AUTHORS = "Songlin Yang, Yu Zhang, Zhiyuan Li"

SKIP_DIRS = {"third_party", "build", "dist", ".eggs", "node_modules"}

HEADER_EXACT = {
    "# -*- coding: utf-8 -*-",
    "#",
    "",
    "# This source code is licensed under the MIT license found in the",
    "# LICENSE file in the root directory of this source tree.",
    "# For a list of all contributors, visit:",
    f"#   {REPO_URL}/graphs/contributors",
}


def make_header(end_year: int) -> str:
    return (
        f"# Copyright (c) {START_YEAR}-{end_year}, {AUTHORS}\n"
        f"#\n"
        f"# This source code is licensed under the MIT license found in the\n"
        f"# LICENSE file in the root directory of this source tree.\n"
        f"# For a list of all contributors, visit:\n"
        f"#   {REPO_URL}/graphs/contributors\n"
    )


HEADER_LINES = 6


def should_skip(path: Path) -> bool:
    return any(d in path.parts for d in SKIP_DIRS)


def is_header_line(line: str) -> bool:
    if line in HEADER_EXACT:
        return True
    # Only match project copyright, not third-party
    return line.startswith("# Copyright (c)") and AUTHORS.split(",")[0] in line


def strip_old_header(lines: list[str]) -> list[str]:
    """Remove all project header lines from the top of the file."""
    i = 0
    while i < len(lines) and is_header_line(lines[i]):
        i += 1
    return lines[i:]


def check_or_fix(root: Path, fix: bool, end_year: int) -> list[str]:
    expected_header = make_header(end_year)
    expected_lines = expected_header.rstrip("\n").split("\n")
    problems = []

    for fpath in sorted(root.rglob("*.py")):
        rel = fpath.relative_to(root)
        if should_skip(rel):
            continue

        content = fpath.read_text(encoding="utf-8", errors="replace")

        if not content.strip():
            continue

        lines = content.split("\n")

        # Already correct
        if lines[:HEADER_LINES] == expected_lines:
            # Verify no duplicate of *our* header right after
            rest = lines[HEADER_LINES:]
            has_dup = any(
                rest[i].startswith("# Copyright (c)") and AUTHORS.split(",")[0] in rest[i]
                for i in range(min(5, len(rest)))
            )
            if not has_dup:
                continue

        problems.append(str(rel))

        if not fix:
            continue

        # Strip all old header lines, then prepend the correct one
        body = strip_old_header(lines)
        # Remove leading blank lines
        while body and body[0] == "":
            body.pop(0)
        new_content = expected_header + "\n" + "\n".join(body)
        # Ensure file ends with newline
        if not new_content.endswith("\n"):
            new_content += "\n"
        fpath.write_text(new_content, encoding="utf-8")

    return problems


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--fix", action="store_true")
    parser.add_argument("--year", type=int, default=datetime.datetime.now().year)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent

    problems = check_or_fix(root, fix=args.fix, end_year=args.year)

    if args.fix:
        if problems:
            print(f"Fixed {len(problems)} file(s):")
            for p in problems:
                print(f"  {p}")
        else:
            print("All files already have correct headers.")
        return 0

    if problems:
        print(f"Found {len(problems)} file(s) with missing or incorrect headers:")
        for p in problems:
            print(f"  {p}")
        print("\nRun 'python scripts/check_header.py --fix' to fix them.")
        return 1

    print("All headers OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
