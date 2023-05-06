"""
Copyright © 2021 yu9824
"""

from typing import overload, Union, Optional, Generator

# The fllowing has deprecated in Python >= 3.9
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
    def __init__(
        self,
        n_splits: int = 5,
        *,
        metric: str = "euclidean",
        n_jobs: Optional[int] = None,
    ) -> None:
        pass

    def __init__(
        self,
        n_splits: int = 5,
        *,
        metric: str = "euclidean",
        n_jobs: Optional[int] = None,
        random_state: None = None,
        shuffle: None = None,
    ) -> None:
        """K-Folds cross-validator using the Kennard-Stone algorithm.

        Parameters
        ----------
        n_splits : int, optional
            Number of folds. Must be at least 2., by default 5

        metric : str, optional
            The distance metric to use. See the documentation of
            `sklearn.metrics.pairwise_distances` for valid values.
            , by default "euclidean"

            =============== ========================================
            metric          Function
            =============== ========================================
            'cityblock'     metrics.pairwise.manhattan_distances
            'cosine'        metrics.pairwise.cosine_distances
            'euclidean'     metrics.pairwise.euclidean_distances
            'haversine'     metrics.pairwise.haversine_distances
            'l1'            metrics.pairwise.manhattan_distances
            'l2'            metrics.pairwise.euclidean_distances
            'manhattan'     metrics.pairwise.manhattan_distances
            'nan_euclidean' metrics.pairwise.nan_euclidean_distances
            =============== ========================================

        n_jobs : int, optional
            The number of parallel jobs., by default None
        """
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.metric = metric
        self.n_jobs = n_jobs

        if shuffle is not None:
            warnings.warn(
                "`shuffle` is unnecessary because it is always shuffled"
                " in this algorithm.",
                UserWarning,
            )
        del self.shuffle

        if random_state is not None:
            warnings.warn(
                "`random_state` is unnecessary since it is uniquely determined"
                " in this algorithm.",
                UserWarning,
            )
        del self.random_state

    def _iter_test_indices(
        self, X=None, y=None, groups=None
    ) -> Generator[List[int], None, None]:
        ks = _KennardStone(
            n_groups=self.get_n_splits(),
            scale=True,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        indexes = ks.get_indexes(X)

        for index in indexes:
            yield index


class KSSplit(BaseShuffleSplit):
    def __init__(
        self,
        n_splits: int = 1,
        *,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        metric: str = "euclidean",
        n_jobs: Optional[int] = None,
    ):
        super().__init__(
            n_splits=n_splits, test_size=test_size, train_size=train_size
        )
        self.metric = metric
        self.n_jobs = n_jobs

        assert self.get_n_splits() == 1, "n_splits must be 1"
        self._default_test_size = 0.1

    # overwrap abstractmethod
    def _iter_indices(self, X, y=None, groups=None):
        ks = _KennardStone(
            n_groups=1, scale=True, metric=self.metric, n_jobs=self.n_jobs
        )
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
    train_size: Optional[Union[float, int]] = None,
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
) -> list:
    pass


def train_test_split(
    *arrays,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
    random_state: None = None,
    shuffle: None = None,
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

    metric : str, optional
        The distance metric to use. See the documentation of
        `sklearn.metrics.pairwise_distances` for valid values.
        , by default "euclidean"

        =============== ========================================
        metric          Function
        =============== ========================================
        'cityblock'     metrics.pairwise.manhattan_distances
        'cosine'        metrics.pairwise.cosine_distances
        'euclidean'     metrics.pairwise.euclidean_distances
        'haversine'     metrics.pairwise.haversine_distances
        'l1'            metrics.pairwise.manhattan_distances
        'l2'            metrics.pairwise.euclidean_distances
        'manhattan'     metrics.pairwise.manhattan_distances
        'nan_euclidean' metrics.pairwise.nan_euclidean_distances
        =============== ========================================

    n_jobs : int, optional
        The number of parallel jobs., by default None

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs

    Raises
    ------
    ValueError
    """
    if shuffle is not None:
        warnings.warn(
            "`shuffle` is unnecessary because it is always shuffled"
            " in this algorithm.",
            UserWarning,
        )

    if random_state is not None:
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
    cv = CVClass(
        test_size=n_test, train_size=n_train, metric=metric, n_jobs=n_jobs
    )

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

        metric : str, optional
            The distance metric to use. See the documentation of
            `sklearn.metrics.pairwise_distances` for valid values.
            , by default "euclidean"

            =============== ========================================
            metric          Function
            =============== ========================================
            'cityblock'     metrics.pairwise.manhattan_distances
            'cosine'        metrics.pairwise.cosine_distances
            'euclidean'     metrics.pairwise.euclidean_distances
            'haversine'     metrics.pairwise.haversine_distances
            'l1'            metrics.pairwise.manhattan_distances
            'l2'            metrics.pairwise.euclidean_distances
            'manhattan'     metrics.pairwise.manhattan_distances
            'nan_euclidean' metrics.pairwise.nan_euclidean_distances
            =============== ========================================

        n_jobs : int, optional
            The number of parallel jobs., by default None
        """
        self.n_groups = n_groups
        self.scale = scale
        self.metric = metric
        self.n_jobs = n_jobs

    def get_indexes(self, X) -> List[List[int]]:
        # check input array
        X: np.ndarray = check_array(
            X,
            ensure_2d=True,
            dtype="numeric",
            force_all_finite="allow-nan"
            if self.metric == "nan_euclidean"
            else True,
        )
        n_samples = X.shape[0]

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

        distance_min = self.distance_matrix[idx_farthest, :]

        # params
        indexes_selected = idx_farthest
        lst_indexes_selected_prev = [[] for _ in range(self.n_groups)]
        indexes_remaining_prev = list(range(n_samples))

        for _ in range(
            n_samples // self.n_groups + bool(n_samples % self.n_groups) - 1
        ):
            # collect the current indexes
            indexes_remaining: List[int] = list()
            arg_selected: List[int] = list()
            for j, idx in enumerate(indexes_remaining_prev):
                if idx in set(indexes_selected):
                    arg_selected.append(j)
                else:
                    indexes_remaining.append(idx)
            n_remaining = len(indexes_remaining)

            lst_indexes_selected = [
                indexes_selected_prev + [index_selected]
                for indexes_selected_prev, index_selected in zip(
                    lst_indexes_selected_prev, indexes_selected
                )
            ]
            # /collect the current indexes

            # 代表長さを決定する
            distance_selected: np.ndarray = self.distance_matrix[
                np.ix_(indexes_selected, indexes_remaining)
            ]
            distance_min = np.delete(distance_min, arg_selected, axis=1)

            distance_min: np.ndarray = np.min(
                np.concatenate(
                    [
                        distance_selected.reshape(self.n_groups, 1, -1),
                        distance_min.reshape(self.n_groups, 1, -1),
                    ],
                    axis=1,
                ),
                axis=1,
            )

            # まだ選択されていない各サンプルにおいて、これまで選択されたすべてのサンプルとの間で
            # ユークリッド距離を計算し，その最小の値を「代表長さ」とする．

            _st_arg_delete: Set[int] = set()
            indexes_selected_next: List[int] = list()
            for k in range(self.n_groups):
                if k == 0:
                    arg_delete = np.argmax(
                        distance_min[k],
                    )
                elif 0 < n_remaining - k:
                    sorted_args = np.argsort(
                        distance_min[k],
                    )
                    # 最大値を取るサンプル (代表長さが最も大きい) のindex_numberを保存
                    for j in range(n_remaining - k, -1, -1):
                        arg_delete = sorted_args[j]
                        if arg_delete not in _st_arg_delete:
                            break
                else:
                    break

                _st_arg_delete.add(arg_delete)
                index_selected: int = indexes_remaining[arg_delete]

                indexes_selected_next.append(index_selected)

            indexes_selected = indexes_selected_next
            lst_indexes_selected_prev = lst_indexes_selected
            indexes_remaining_prev = indexes_remaining
        else:  # もうないなら遠い順から近い順 (test側) に並べ替えて終える
            assert n_remaining - len(indexes_selected_next) <= 0
            indexes_output: List[List[int]] = []
            for k in range(self.n_groups):
                indexes_selected_reversed = lst_indexes_selected[k][::-1]
                if k < len(indexes_selected_next):
                    index_selected_next = indexes_selected_next[k]
                    indexes_output.append(
                        [index_selected_next] + indexes_selected_reversed
                    )
                else:
                    indexes_output.append(indexes_selected_reversed)
            return indexes_output


if __name__ == "__main__":
    from sklearn.model_selection import cross_validate
    from sklearn.datasets import load_diabetes, fetch_california_housing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    data = fetch_california_housing(as_frame=True)
    # data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target

    # ks = _KennardStone(n_groups=2, scale=True, n_jobs=-1)
    # ks = _KennardStone(n_groups=1, scale=True, n_jobs=-1)
    # ks.get_indexes(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, n_jobs=-1
    )
    rf = RandomForestRegressor(n_jobs=-1, random_state=334)
    rf.fit(X_train, y_train)
    y_pred_on_test = rf.predict(X_test)
    print(mean_squared_error(y_test, y_pred_on_test, squared=False))

    # kf = KFold(n_splits=5, n_jobs=-1)
    # print(cross_validate(rf, X, y, scoring="neg_mean_squared_error", cv=kf))
