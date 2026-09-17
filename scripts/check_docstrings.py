#!/usr/bin/env python3
"""Check python/ docstrings for design-rationale language.

Docstrings state the contract; they must not narrate the process that led
to it ("deliberately", "used to", "originally", "rather than", "instead
of", "replaces", an issue/PR number) — that belongs in the commit message,
not the code.

Run: python3 scripts/check_docstrings.py
"""

import ast
import re
import sys
from pathlib import Path
from typing import Iterator, NamedTuple

SRC = Path(__file__).resolve().parent.parent / "python"

BANNED_PATTERN = re.compile(
    r"issue #|#\d+|deliberately|used to|originally|rather than|instead of|replaces",
    re.IGNORECASE,
)


class Violation(NamedTuple):
    path: Path
    line: int
    detail: str


def _docstring_violations(tree: ast.Module, path: Path) -> Iterator[Violation]:
    """Banned-language violations on every docstring in the file, including
    nested closures, so rationale can't hide there."""
    nodes = [(tree, 1)] + [
        (node, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for node, lineno in nodes:
        docstring = ast.get_docstring(node, clean=False)
        if docstring is None:
            continue
        match = BANNED_PATTERN.search(docstring)
        if match:
            name = getattr(node, "name", "<module>")
            yield Violation(path, lineno, f"{name!r} docstring matches {match.group()!r}")


def check_file(path: Path) -> list[Violation]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return list(_docstring_violations(tree, path))


def main() -> int:
    violations = []
    for path in sorted(SRC.rglob("*.py")):
        violations.extend(check_file(path))

    if not violations:
        print("check_docstrings: OK")
        return 0

    for v in violations:
        rel = v.path.relative_to(SRC.parent)
        print(f"{rel}:{v.line}: {v.detail}")
    print(f"\ncheck_docstrings: {len(violations)} violation(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
