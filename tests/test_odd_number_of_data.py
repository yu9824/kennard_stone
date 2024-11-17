from sklearn.datasets import load_diabetes

from kennard_stone._core._core import _KennardStone


def test_odd_number_of_data():
    diabetes = load_diabetes(as_frame=True)

    X = diabetes.data
    _KennardStone(n_groups=2).get_indexes(X)

    X = X.iloc[:-1]
    _KennardStone(n_groups=2).get_indexes(X)
