"""
Copyright © 2021 yu9824
"""

from typing import overload, Union, Optional

# deprecated in Python >= 3.9
from typing import List, Set
from itertools import chain
import warnings

import numpy as np

from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils.validation import _num_samples
from sklearn.utils import indexable
from sklearn.utils import _safe_indexing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_array


# TODO: sphinx documentation？
# TODO: parallelization


class KFold(_BaseKFold):
    @overload
    def __init__(self, n_splits: int = 5):
        pass

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


@overload
def train_test_split(
    *arrays,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None
) -> list:
    pass


def train_test_split(
    *arrays,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    **kwargs
) -> list:
    """Split arrays or matrices into train and test subsets using the
    Kennard-Stone algorithm.

    Data partitioning by the Kennard-Stone algorithm is performed based on the
     first element to be input.

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
    if "shuffle" in kwargs:
        warnings.warn(
            "`shuffle` is unnecessary because it is always shuffled"
            " in this algorithm.",
            UserWarning,
        )

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
    def __init__(
        self,
        n_groups: int = 1,
        scale: bool = True,
        metric: str = "euclidean",
        n_jobs: Optional[int] = None,
    ) -> None:
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
        self.metric = metric
        self.n_jobs = n_jobs

    def get_indexes(self, X) -> List[List[int]]:
        # check input array
        X: np.ndarray = check_array(X, ensure_2d=True, dtype="numeric")

        # drop no variance
        vselector = VarianceThreshold(threshold=0.0)
        X = vselector.fit_transform(X)

        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Save the original X.
        # self._original_X = X.copy()

        # Pre-calculate the distance matrix.
        self.distance_matrix = pairwise_distances(
            X, metric=self.metric, n_jobs=self.n_jobs
        )

        # 全ての組成に対してそれぞれの平均との距離の二乗を配列として得る． (サンプル数の分だけ存在)
        # distance_to_ave = np.sum(np.square(X - X.mean(axis=0)), axis=1)
        distance_to_ave = pairwise_distances(
            X,
            X.mean(axis=0, keepdims=True),
            metric=self.metric,
            n_jobs=self.n_jobs,
        ).flatten()

        # 最大値を取るサンプル (平均からの距離が一番遠い) のindex_numberを保存
        idx_farthest: List[int] = np.argsort(distance_to_ave)[::-1][
            : self.n_groups
        ].tolist()

        # 抜き出した (train用) サンプルのindex_numberを保存しとくリスト
        lst_indexes_selected: List[List[int]] = [
            [_idx] for _idx in idx_farthest
        ]

        # まだ抜き出しておらず，残っているサンプル (test用) サンプルのindex_numberを保存しておくリスト
        indexes_remaining: List[int] = [
            _idx for _idx in range(len(X)) if _idx not in set(idx_farthest)
        ]

        # 近い順のindexのリスト．i.e. 最初がtest向き，最後がtrain向き
        indexes = self._sort(
            lst_indexes_selected=lst_indexes_selected,
            indexes_remaining=indexes_remaining,
        )

        assert (
            len(tuple(chain.from_iterable(indexes)))
            == len(set(chain.from_iterable(indexes)))
            == len(X)
        )

        return indexes

    def _sort(
        self,
        lst_indexes_selected: List[List[int]],
        indexes_remaining: List[int],
    ) -> List[List[int]]:

        idx_selected: List[int] = list(
            chain.from_iterable(lst_indexes_selected)
        )
        min_distance_to_samples_selected: np.ndarray = self.distance_matrix[
            np.ix_(idx_selected, indexes_remaining)
        ].reshape(
            self.n_groups,
            len(idx_selected) // self.n_groups,
            -1,  # len(indexes_remaining)
        )

        # まだ選択されていない各サンプルにおいて、これまで選択されたすべてのサンプルとの間で
        # ユークリッド距離を計算し，その最小の値を「代表長さ」とする．

        _st_i_delete: Set[int] = set()
        for k in range(self.n_groups):
            if k == 0:
                i_deleted = np.argmax(
                    np.min(min_distance_to_samples_selected[k], axis=0)
                )
            elif 0 < len(indexes_remaining) - k:
                _lst_sorted_args = np.argsort(
                    np.min(
                        min_distance_to_samples_selected[k],
                        axis=0,
                    ),
                )
                j = len(indexes_remaining) - 1
                while _lst_sorted_args[j] in _st_i_delete:
                    j -= 1
                else:
                    # 最大値を取るサンプル (代表長さが最も大きい) のindex_numberを保存
                    i_deleted = _lst_sorted_args[j]
            else:
                break

            idx_selected: int = indexes_remaining[i_deleted]

            lst_indexes_selected[k].append(idx_selected)
            _st_i_delete.add(i_deleted)

        # delete
        indexes_remaining = [
            _idx
            for i, _idx in enumerate(indexes_remaining)
            if i not in _st_i_delete
        ]

        if len(indexes_remaining):  # まだ残っているなら再帰
            return self._sort(lst_indexes_selected, indexes_remaining)
        else:  # もうないなら遠い順から近い順 (test側) に並べ替えて終える
            return [
                _idx_selected[::-1] for _idx_selected in lst_indexes_selected
            ]


if __name__ == "__main__":
    from sklearn.model_selection import cross_validate
    from sklearn.datasets import load_diabetes, fetch_california_housing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    # data = fetch_california_housing(as_frame=True)
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    ks = _KennardStone(n_groups=2, scale=True)
    ks.get_indexes(X)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # rf = RandomForestRegressor(n_jobs=-1, random_state=334)
    # rf.fit(X_train, y_train)
    # y_pred_on_test = rf.predict(X_test)
    # print(mean_squared_error(y_test, y_pred_on_test, squared=False))

    # kf = KFold(n_splits=5)
    # print(cross_validate(rf, X, y, scoring="neg_mean_squared_error", cv=kf))
