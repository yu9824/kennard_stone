"""The selection order must not change from v3.0.2.

The only exception is where v3.0.2 assigned a sample to two groups (and
another sample to none). Such outputs were fixed in v3.1.0, so they only
need to be valid partitions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _legacy_v3_0_2 import _KennardStone as _LegacyKennardStone
from _regression_cases import (
    CORE_CASES,
    DATASETS,
    EXACT_CASES,
    has_ties,
    is_partition,
    run_core_case,
)

from kennard_stone import KFold, train_test_split

SNAPSHOT = json.loads(
    (Path(__file__).parent / "data" / "regression_snapshot.json").read_text()
)


def _ids(cases):
    return [name for name, _, _ in cases]


@pytest.mark.parametrize(
    ("name", "dataset", "kwargs"), CORE_CASES, ids=_ids(CORE_CASES)
)
def test_core_matches_legacy(name, dataset, kwargs):
    result = run_core_case(dataset, kwargs)
    legacy = run_core_case(dataset, kwargs, cls=_LegacyKennardStone)
    n_samples = len(DATASETS[dataset]())

    assert is_partition(result, n_samples)
    if is_partition(legacy, n_samples):
        assert result == legacy


@pytest.mark.parametrize(
    ("name", "dataset", "kwargs"), EXACT_CASES, ids=_ids(EXACT_CASES)
)
def test_exact_cases_match_snapshot(name, dataset, kwargs):
    # The snapshot is portable only if no tie-breaking is involved.
    assert not has_ties(DATASETS[dataset](), kwargs.get("metric", "euclidean"))
    assert run_core_case(dataset, kwargs) == SNAPSHOT["core"][name]


@pytest.mark.parametrize(
    ("test_size", "n_test"), [(0.1, 45), (0.25, 111), (7, 7)]
)
def test_train_test_split_matches_legacy(test_size, n_test):
    X = DATASETS["diabetes"]()
    (order,) = run_core_case("diabetes", {}, cls=_LegacyKennardStone)

    _, _, train, test = train_test_split(
        X, np.arange(len(X)), test_size=test_size
    )

    assert test.tolist() == order[:n_test]
    assert train.tolist() == order[n_test:]


@pytest.mark.parametrize("n_splits", [2, 5])
def test_kfold_matches_legacy(n_splits):
    X = DATASETS["diabetes"]()
    folds = run_core_case(
        "diabetes", {"n_groups": n_splits}, cls=_LegacyKennardStone
    )

    # scikit-learn returns the indexes of each split in ascending order.
    for (train, test), fold in zip(KFold(n_splits=n_splits).split(X), folds):
        assert test.tolist() == sorted(fold)
        assert train.tolist() == sorted(
            i for other in folds if other is not fold for i in other
        )
