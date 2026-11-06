"""Compare init="farthest_pair" with an independent reference implementation
of the original Kennard-Stone algorithm (Kennard & Stone, 1969)."""

from __future__ import annotations

import numpy as np
import pytest
from _regression_cases import DATASETS
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from kennard_stone._core._core import _KennardStone


def kennard_stone_reference(X: np.ndarray, scale: bool = True) -> list[int]:
    """Plain implementation of the original algorithm.

    Start from the two most distant samples, then repeatedly add the sample
    whose distance to its nearest selected sample is the largest. Ties are
    broken by the smallest index.
    """
    X = VarianceThreshold(threshold=0.0).fit_transform(X)
    if scale:
        X = StandardScaler().fit_transform(X)
    D = pairwise_distances(X)

    i, j = np.unravel_index(np.argmax(D), D.shape)
    selected = [int(i), int(j)]
    min_dist = np.minimum(D[i], D[j])
    remaining = np.ones(len(X), dtype=bool)
    remaining[selected] = False

    while remaining.any():
        candidates = np.flatnonzero(remaining)
        k = int(candidates[np.argmax(min_dist[candidates])])
        selected.append(k)
        remaining[k] = False
        min_dist = np.minimum(min_dist, D[k])
    return selected


def _selection_order(X: np.ndarray, scale: bool = True) -> list[int]:
    (indexes,) = _KennardStone(init="farthest_pair", scale=scale).get_indexes(
        X
    )
    # get_indexes returns indexes in reverse order of selection.
    return [int(i) for i in indexes[::-1]]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("distribution", ["normal", "lognormal", "clustered"])
def test_matches_reference_random(seed, distribution):
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        X = rng.normal(size=(60, 3))
    elif distribution == "lognormal":
        X = rng.lognormal(size=(60, 3))
    else:
        X = np.concatenate(
            [rng.normal(size=(50, 2)), rng.normal(5, 0.1, size=(10, 2))]
        )

    assert _selection_order(X) == kennard_stone_reference(X)


@pytest.mark.parametrize("dataset", ["diabetes", "diabetes_dup", "grid"])
@pytest.mark.parametrize("scale", [True, False])
def test_matches_reference_datasets(dataset, scale):
    X = DATASETS[dataset]()
    assert _selection_order(X, scale=scale) == kennard_stone_reference(
        X, scale=scale
    )
