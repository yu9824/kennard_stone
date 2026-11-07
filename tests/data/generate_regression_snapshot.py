"""Regenerate ``regression_snapshot.json`` (EXACT_CASES only).

Run only when the selection order is intentionally changed::

    PYTHONPATH=src python tests/data/generate_regression_snapshot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Imported after extending sys.path above.
from _regression_cases import EXACT_CASES, run_core_case  # noqa: E402

import kennard_stone  # noqa: E402


def main() -> None:
    snapshot = {
        "kennard_stone_version": kennard_stone.__version__,
        "core": {
            name: run_core_case(dataset, kwargs)
            for name, dataset, kwargs in EXACT_CASES
        },
    }
    path = HERE / "regression_snapshot.json"
    path.write_text(json.dumps(snapshot, separators=(",", ":")) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
