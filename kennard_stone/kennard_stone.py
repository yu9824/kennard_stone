"""
Copyright © 2021 yu9824
"""

from typing import List, Union, Optional
import warnings

import numpy as np

from itertools import chain
from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils.validation import _num_samples
from sklearn.utils import indexable, _safe_indexing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array


# TODO: unittest?
# TODO: sphinx documentation？

class KFold(_BaseKFold):
    def __init__(self, n_splits: int = 5, **kwargs):
        """K-Folds cross-validator using the Kennard-Stone algorithm.

        Parameters
        ----------
        n_splits : int, optional
            Number of folds. Must be at least 2., by default 5
        """
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)

        if "shuffle" in kwargs:
            warnings.warn(
                "`shuffle` is unnecessary because it is always shuffled"
                " in this algorithm.",
                UserWarning,
            )
        del self.shuffle

        if "random_state" in kwargs:
            warnings.warn(
                "`random_state` is unnecessary since it is uniquely determined"
                " in this algorithm.",
                UserWarning,
            )
        del self.random_state

    def _iter_test_indices(self, X=None, y=None, groups=None):
        ks = _KennardStone(n_groups=self.get_n_splits())
        indexes = ks.get_indexes(X)

        for index in indexes:
            yield index


class KSSplit(BaseShuffleSplit):
    def __init__(
        self,
        n_splits: int = 1,
        *,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None
    ):
        super().__init__(
            n_splits=n_splits, test_size=test_size, train_size=train_size
        )
        assert self.get_n_splits() == 1, "n_splits must be 1"
        self._default_test_size = 0.1

    # overwrap abstractmethod
    def _iter_indices(self, X, y=None, groups=None):
        ks = _KennardStone(n_groups=1)
        indexes = ks.get_indexes(X)[0]

        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        for _ in range(self.get_n_splits()):
            ind_test = indexes[:n_test]
            ind_train = indexes[n_test : (n_test + n_train)]
            yield ind_train, ind_test


def train_test_split(*arrays, test_size=None, train_size=None, **kwargs):
    """Split arrays or matrices into train and test subsets using the
    Kennard-Stone algorithm.

    Parameters
    ----------
    *arrays: sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If train_size is also None, it will be
        set to 0.25., by default None
    train_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split. If int, represents the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size., by default None

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs

    Raises
    ------
    ValueError
    """
    if "random_state" in kwargs:
        warnings.warn(
            "`random_state` is unnecessary since it is uniquely determined"
            " in this algorithm.",
            UserWarning,
        )

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    CVClass = KSSplit
    cv = CVClass(test_size=n_test, train_size=n_train)

    train, test = next(cv.split(X=arrays[0]))

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


class _KennardStone:
    def __init__(self, n_groups: int = 1, scale: bool = True) -> None:
        """The root program of the Kennard-Stone algorithm.

        Parameters
        ----------
        n_groups : int, optional
            how many groups to divide, by default 1
        scale : bool, optional
            scaling X or not, by default True
        """
        self.n_groups = n_groups
        self.scale = scale

    def get_indexes(self, X) -> List[List[int]]:
        # check input array
        X: np.ndarray = check_array(X, ensure_2d=True)

        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Save the original X.
        self._original_X = X.copy()

        # 全ての組成に対してそれぞれの平均との距離の二乗を配列として得る． (サンプル数の分だけ存在)
        distance_to_ave = np.sum(np.square(X - X.mean(axis=0)), axis=1)

        # 最大値を取るサンプル (平均からの距離が一番遠い) のindex_numberを保存
        idx_farthest = np.argsort(distance_to_ave)[::-1][: self.n_groups]

        # 抜き出した (train用) サンプルのindex_numberを保存しとくリスト
        lst_idx_selected: List[List[int]] = [[_idx] for _idx in idx_farthest]

        # まだ抜き出しておらず，残っているサンプル (test用) サンプルのindex_numberを保存しておくリスト
        idx_remaining = np.arange(len(X))

        # 抜き出した (train用) サンプルに選ばれたサンプルをtrain用のものから削除
        X = np.delete(X, idx_farthest, axis=0)
        idx_remaining = np.delete(idx_remaining, idx_farthest, axis=0)

        # 近い順のindexのリスト．i.e. 最初がtest向き，最後がtrain向き
        indexes = self._sort(
            X=X, lst_idx_selected=lst_idx_selected, idx_remaining=idx_remaining
        )
        assert (
            len(sum(indexes, start=[]))
            == len(set(sum(indexes, start=[])))
            == len(self._original_X)
        )

        return indexes

    def _sort(
        self,
        X,
        lst_idx_selected: List[List[int]],
        idx_remaining: Union[List[int], np.ndarray],
    ) -> List[List[int]]:
        samples_selected: np.ndarray = self._original_X[
            sum(lst_idx_selected, start=[])
        ]

        # まだ選択されていない各サンプルにおいて、これまで選択されたすべてのサンプルとの間で
        # ユークリッド距離を計算し，その最小の値を「代表長さ」とする．

        min_distance_to_samples_selected = np.sum(
            np.square(
                np.expand_dims(samples_selected, 1) - np.expand_dims(X, 0)
            ),
            axis=2,
        )

        _idxes_delete: List[int] = []
        n_selected = len(lst_idx_selected[0])
        for k in range(len(lst_idx_selected)):
            if 0 < len(idx_remaining) - k:
                _lst_sorted_args = np.argsort(
                    np.min(
                        min_distance_to_samples_selected[
                            n_selected * k : n_selected * (k + 1)
                        ],
                        axis=0,
                    ),
                )
                j = len(idx_remaining) - 1
                while _lst_sorted_args[j] in set(_idxes_delete):
                    j -= 1
                else:
                    # 最大値を取るサンプル (代表長さが最も大きい) のindex_numberを保存
                    idx_selected = _lst_sorted_args[j]

                lst_idx_selected[k].append(idx_remaining[idx_selected])
                _idxes_delete.append(idx_selected)
            else:
                break

        # delete
        X = np.delete(X, _idxes_delete, axis=0)
        idx_remaining = np.delete(idx_remaining, _idxes_delete, axis=0)

        if len(idx_remaining):  # まだ残っているなら再帰
            return self._sort(X, lst_idx_selected, idx_remaining)
        else:  # もうないなら遠い順から近い順 (test側) に並べ替えて終える
            return [_idx_selected[::-1] for _idx_selected in lst_idx_selected]


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import cross_validate
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error as mse

    data = load_diabetes(as_frame=True)
    X: pd.DataFrame = data.data
    y: pd.Series = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf = RandomForestRegressor(n_jobs=-1, random_state=334)
    rf.fit(X_train, y_train)
    print(mse(rf.predict(X_test), y_test))

    kf = KFold(n_splits=5)
    print(cross_validate(rf, X, y, scoring="neg_mean_squared_error", cv=kf))
