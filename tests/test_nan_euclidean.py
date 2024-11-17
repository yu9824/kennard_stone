from sklearn.datasets import load_diabetes

from kennard_stone._core._core import _KennardStone


def test_nan_euclidean():
    X = load_diabetes(as_frame=True).data.copy()

    X.iloc[1, 1] = float("nan")

    ks = _KennardStone(n_groups=1, metric="nan_euclidean")
    ks.get_indexes(X)
