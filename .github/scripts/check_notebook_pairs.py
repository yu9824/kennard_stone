"""Check that each notebook matches its paired py:percent file.

Only the cell types and sources are compared, so the outputs and the
jupytext version written in the header do not matter.

Usage: python .github/scripts/check_notebook_pairs.py docs_src
"""

from __future__ import annotations

import sys
from pathlib import Path

import jupytext


def _cells(path: Path) -> list[tuple[str, str]]:
    return [
        (cell.cell_type, cell.source.strip())
        for cell in jupytext.read(path).cells
    ]


def main(root: str) -> int:
    """Return 1 if any notebook under ``root`` differs from its pair."""
    errors = []
    for ipynb in sorted(Path(root).rglob("*.ipynb")):
        if ".ipynb_checkpoints" in ipynb.parts:
            continue
        py = ipynb.with_suffix(".py")
        if not py.is_file():
            errors.append(f"{py} is missing")
        elif _cells(ipynb) != _cells(py):
            errors.append(f"{ipynb} and {py} differ")
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(
            "Run `jupytext --sync` on the notebooks and commit both files.",
            file=sys.stderr,
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
