"""Every sample must belong to exactly one group (one test fold)."""

from __future__ import annotations

import numpy as np
import pytest
from _regression_cases import DATASETS

from kennard_stone import KFold
from kennard_stone._core._core import _KennardStone


def _random_data(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n_samples = rng.randint(6, 60)
    if seed % 2:
        # Integer data with many tied distances.
        return rng.randint(0, 5, size=(n_samples, 2)).astype(float)
    return rng.rand(n_samples, 3)


@pytest.mark.parametrize("n_splits", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(20))
def test_kfold_test_folds_partition_samples(seed, n_splits):
    X = _random_data(seed)

    test_folds = [test for _, test in KFold(n_splits=n_splits).split(X)]

    assert len(test_folds) == n_splits
    assert sorted(np.concatenate(test_folds).tolist()) == list(range(len(X)))


def test_three_groups_on_small_data():
    # v3.0.2 assigned sample 6 to the 1st and 3rd groups and sample 5 to
    # none. v3.1.0 fixed that ([[6, 10, 3, 11], [4, 7, 2, 1], [5, 8, 9, 0]]),
    # and v4.0.0 lets the 3rd group take its best free sample (e.g. 10
    # instead of 9 in the first round).
    X = DATASETS["distinct"]()

    groups = _KennardStone(n_groups=3, scale=False).get_indexes(X)

    assert [list(g) for g in groups] == [
        [6, 9, 3, 11],
        [8, 7, 2, 1],
        [4, 5, 10, 0],
    ]
