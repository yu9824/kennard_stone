import pytest
from sklearn.datasets import load_diabetes

from kennard_stone import KFold
from kennard_stone._core._core import _KennardStone


def test_odd_number_of_data():
    X, _ = load_diabetes(return_X_y=True)

    _KennardStone(n_groups=2).get_indexes(X)

    X = X[:-1]
    _KennardStone(n_groups=2).get_indexes(X)


def test_as_many_groups_as_samples():
    X, _ = load_diabetes(return_X_y=True)
    X = X[:4]

    indexes = _KennardStone(n_groups=4).get_indexes(X)

    assert all(len(group) == 1 for group in indexes)
    assert sorted(i for group in indexes for i in group) == [0, 1, 2, 3]


def test_kfold_leave_one_out_like():
    X, _ = load_diabetes(return_X_y=True)
    X = X[:5]

    tests = [test.tolist() for _, test in KFold(n_splits=5).split(X)]

    assert sorted(tests) == [[0], [1], [2], [3], [4]]


def test_more_groups_than_samples():
    X, _ = load_diabetes(return_X_y=True)

    with pytest.raises(ValueError, match="must not exceed"):
        _KennardStone(n_groups=4).get_indexes(X[:3])
