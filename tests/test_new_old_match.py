from sklearn.datasets import load_diabetes

from kennard_stone import _deprecated
from kennard_stone._core import _core


def test_new_old_match():
    X, _ = load_diabetes(return_X_y=True)

    ks_old = _deprecated._KennardStone()
    ks_new = _core._KennardStone(n_groups=1)

    assert ks_old._get_indexes(X) == ks_new.get_indexes(X)[0].tolist()
