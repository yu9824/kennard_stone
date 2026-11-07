"""Tests for options of the public API."""

from __future__ import annotations

import numpy as np
import pytest
from _legacy_v3_0_2 import _KennardStone as _LegacyKennardStone
from _regression_cases import DATASETS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from kennard_stone import KFold, train_test_split
from kennard_stone._core._core import _KennardStone


@pytest.fixture
def X() -> np.ndarray:
    # Features on very different scales, so scaling changes the order.
    X = DATASETS["diabetes"]()
    return X * np.arange(1, X.shape[1] + 1) ** 3


@pytest.mark.parametrize("scale", [True, False])
def test_train_test_split_scale(X, scale):
    (order,) = [
        list(map(int, indexes))
        for indexes in _LegacyKennardStone(scale=scale).get_indexes(X)
    ]

    _, _, train, test = train_test_split(
        X, np.arange(len(X)), test_size=0.2, scale=scale
    )

    n_test = len(test)
    assert test.tolist() == order[:n_test]
    assert train.tolist() == order[n_test:]


def test_scale_changes_order(X):
    orders = [
        train_test_split(X, np.arange(len(X)), test_size=0.2, scale=scale)[3]
        for scale in (True, False)
    ]
    assert orders[0].tolist() != orders[1].tolist()


@pytest.mark.parametrize("scale", [True, False])
def test_kfold_scale(X, scale):
    folds = _KennardStone(n_groups=3, scale=scale).get_indexes(X)

    splits = list(KFold(n_splits=3, scale=scale).split(X))

    assert [test.tolist() for _, test in splits] == [
        sorted(map(int, fold)) for fold in folds
    ]


def test_init_default_is_farthest_from_mean(X):
    default = train_test_split(X, np.arange(len(X)), test_size=0.2)[3]
    explicit = train_test_split(
        X, np.arange(len(X)), test_size=0.2, init="farthest_from_mean"
    )[3]
    assert default.tolist() == explicit.tolist()


def test_init_farthest_pair_starts_from_most_distant_pair(X):
    D = pairwise_distances(StandardScaler().fit_transform(X))
    pair = {int(i) for i in np.unravel_index(np.argmax(D), D.shape)}

    train = train_test_split(
        X, np.arange(len(X)), test_size=0.2, init="farthest_pair"
    )[2]

    # train holds samples in reverse order of selection.
    assert set(train[-2:].tolist()) == pair


def test_init_invalid_value(X):
    with pytest.raises(ValueError, match="init must be one of"):
        train_test_split(X, test_size=0.2, init="unknown")


def test_init_farthest_pair_requires_single_group(X):
    with pytest.raises(ValueError, match="n_groups=1"):
        _KennardStone(n_groups=2, init="farthest_pair").get_indexes(X)
