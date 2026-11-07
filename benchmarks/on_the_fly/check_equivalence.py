"""Check that the prototype returns exactly the same indexes as the core.

Usage (from the repository root)::

    PYTHONPATH=src python benchmarks/on_the_fly/check_equivalence.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "tests")]

from _prototype import (  # noqa: E402
    _cpu_distance_rows,
    kennard_stone_on_the_fly,
)
from _regression_cases import CORE_CASES, DATASETS  # noqa: E402
from test_max_min import _check_max_min  # noqa: E402

from kennard_stone._core._core import (  # noqa: E402
    _KennardStone,
    _metric_kwargs,
    _preprocess,
    _validate,
)


def _as_lists(groups) -> list[list[int]]:
    return [[int(i) for i in group] for group in groups]


def _is_max_min(distance_matrix, groups) -> bool:
    try:
        _check_max_min(distance_matrix, groups)
    except AssertionError:
        return False
    return True


def _explain(X, ks, expected, actual, metric, scale) -> str:
    """Describe a difference: distance error and validity of both results."""
    X_processed = _preprocess(_validate(X, metric), scale)
    distance_rows = _cpu_distance_rows(
        X_processed, metric, _metric_kwargs(X_processed, metric)
    )
    n_samples = X_processed.shape[0]
    full = ks.distance_matrix
    error = max(
        float(
            np.max(
                np.abs(
                    full[i]
                    - distance_rows(np.array([i]), np.arange(n_samples))
                )
            )
        )
        for i in range(n_samples)
    )
    return (
        f"max distance difference {error:.1e}; max-min holds: "
        f"current={_is_max_min(full, expected)}, "
        f"prototype={_is_max_min(full, actual)}"
    )


def _compare(X, **kwargs) -> str | None:
    """Return None if both agree, otherwise a description."""
    ks = _KennardStone(**kwargs)
    expected = _as_lists(ks.get_indexes(X))
    actual = _as_lists(kennard_stone_on_the_fly(X, **kwargs))
    if expected == actual:
        return None
    return _explain(
        X,
        ks,
        expected,
        actual,
        kwargs.get("metric", "euclidean"),
        kwargs.get("scale", True),
    )


def main() -> int:
    """Return 1 if any case differs."""
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    failures = []
    n_checked = 0

    for name, dataset, kwargs in CORE_CASES:
        X = DATASETS[dataset]()
        n_checked += 1
        if note := _compare(X, **kwargs):
            failures.append(f"{name}: {note}")
        if kwargs["n_groups"] == 1:
            n_checked += 1
            if note := _compare(X, init="farthest_pair", **kwargs):
                failures.append(f"{name} (farthest_pair): {note}")

    rng = np.random.RandomState(0)
    for seed in range(200):
        n_samples = rng.randint(6, 80)
        if seed % 2:
            X = rng.randint(0, 5, size=(n_samples, 2)).astype(float)
            if np.all(X.var(axis=0) == 0):
                continue
        else:
            X = rng.rand(n_samples, 3)
        n_groups = rng.randint(1, min(n_samples, 6) + 1)
        metric = ("euclidean", "manhattan", "chebyshev", "cosine")[seed % 4]
        n_checked += 1
        if note := _compare(X, n_groups=n_groups, metric=metric):
            failures.append(
                f"random seed={seed} n={n_samples} groups={n_groups} "
                f"metric={metric}: {note}"
            )

    print(f"checked {n_checked} cases, {len(failures)} differ")
    for failure in failures:
        print("  differs:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
