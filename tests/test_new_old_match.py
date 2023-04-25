from sklearn.datasets import load_diabetes

from kennard_stone import kennard_stone
from kennard_stone import _deprecated


def test_new_old_match():
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data

    ks_old = _deprecated._KennardStone()
    ks_new = kennard_stone._KennardStone(n_groups=1)

    assert ks_old._get_indexes(X) == ks_new.get_indexes(X)[0]
