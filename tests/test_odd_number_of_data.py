from sklearn.datasets import load_diabetes

from kennard_stone._core._core import _KennardStone


def test_odd_number_of_data():
    X, _ = load_diabetes(return_X_y=True)

    _KennardStone(n_groups=2).get_indexes(X)

    X = X[:-1]
    _KennardStone(n_groups=2).get_indexes(X)
