from sklearn.datasets import load_diabetes

from kennard_stone._core._core import _KennardStone


def test_nan_euclidean():
    X, _ = load_diabetes(return_X_y=True)

    X[1, 1] = float("nan")

    ks = _KennardStone(n_groups=1, metric="nan_euclidean")
    ks.get_indexes(X)
