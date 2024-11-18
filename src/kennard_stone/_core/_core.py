"""
Copyright © 2021 yu9824
"""

from __future__ import annotations

import sys
import warnings
from array import array
from itertools import chain
from typing import Any, Optional, TypeVar, Union, overload

if sys.version_info >= (3, 9):
    from collections.abc import Callable, Generator
else:
    from typing import Callable, Generator


import numpy as np
from numpy.typing import ArrayLike
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection._split import (
    BaseShuffleSplit,
    _BaseKFold,
    _validate_shuffle_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _safe_indexing, check_array, indexable
from sklearn.utils.validation import _num_samples

from ..utils._pairwise import pairwise_distances
from ..utils._type_alias import Device, Metrics
from ..utils._utils import (
    IgnoredArgumentWarning,
)

# for typing
T = TypeVar("T")

# TODO: Update docstrings


class KFold(_BaseKFold):
    def __init__(
        self,
        n_splits: int = 5,
        *,
        metric: Union[
            Metrics, Callable[[ArrayLike, ArrayLike], np.ndarray]
        ] = "euclidean",
        n_jobs: Optional[int] = None,
        device: Device = "cpu",
        shuffle: None = None,
        random_state: None = None,
    ) -> None:
        """K-Folds cross-validator using the Kennard-Stone algorithm.

        Parameters
        ----------
        n_splits : int, optional
            Number of folds. Must be at least 2., by default 5

        metric : Union[Metrics, Callable[[ArrayLike, ArrayLike], np.ndarray]
            , optional

            The distance metric to use. See the documentation of
            - `scipy.spatial.distance.pdist`
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
            - `sklearn.metrics.pairwise_distances`
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

            for valid values.
            , by default "euclidean"

            Valid values for metric are:

            - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1',
                'l2', 'manhattan']. These metrics support sparse matrix inputs.
                ['nan_euclidean'] but it does not yet support sparse matrices.
            - From scipy.spatial.distance: ['braycurtis', 'canberra',
                'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
                'mahalanobis', 'minkowski', 'rogerstanimoto',
                'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                'sqeuclidean', 'yule'] See the documentation for
                scipy.spatial.distance for details on these metrics.
                These metrics do not support sparse matrix inputs.

            If you want to use GPU when calculating the distance matrix
            ('euclidean', 'manhattan', 'chebyshev' and 'minkowski'),
            you need to install 'pytorch' and set `device` to 'cuda' or 'mps'.

        n_jobs : int, optional
            The number of parallel jobs. It is valid only when CPU is used.
            , by default None

        device : Literal['cpu', 'cuda', 'mps'] or torch.device or str, optional
            , by default 'cpu'

            If you want to use GPU when calculating the distance matrix
            ('euclidean', 'manhattan', 'chebyshev' and 'minkowski'),
            you need to install 'pytorch' and set `device` to 'cuda' or 'mps'.

        random_state : None, deprecated
            This parameter is deprecated and has no effect
            because the algorithm is deterministic.

        shuffle : None, deprecated
            This parameter is deprecated and has no effect
            because the algorithm is deterministic.
        """
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.metric = metric
        self.n_jobs = n_jobs
        self.device = device

        if shuffle is not None:
            warnings.warn(
                "`shuffle` is unnecessary because it is always shuffled"
                " in this algorithm.",
                IgnoredArgumentWarning,
            )
        del self.shuffle

        if random_state is not None:
            warnings.warn(
                "`random_state` is unnecessary since it is uniquely determined"
                " in this algorithm.",
                IgnoredArgumentWarning,
            )
        del self.random_state

    def _iter_test_indices(
        self, X=None, y=None, groups=None
    ) -> Generator[array[int], None, None]:
        ks = _KennardStone(
            n_groups=self.get_n_splits(),
            scale=True,
            metric=self.metric,
            n_jobs=self.n_jobs,
            device=self.device,
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
        metric: Union[
            Metrics, Callable[[ArrayLike, ArrayLike], np.ndarray]
        ] = "euclidean",
        n_jobs: Optional[int] = None,
        device: Device = "cpu",
    ):
        super().__init__(
            n_splits=n_splits, test_size=test_size, train_size=train_size
        )
        self.metric = metric
        self.n_jobs = n_jobs
        self.device = device

        assert self.get_n_splits() == 1, "n_splits must be 1"
        self._default_test_size = 0.1

    # overwrap abstractmethod
    def _iter_indices(
        self, X, y=None, groups=None
    ) -> Generator[tuple[list[int], list[int]], None, None]:
        ks = _KennardStone(
            n_groups=1,
            scale=True,
            metric=self.metric,
            n_jobs=self.n_jobs,
            device=self.device,
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
            ind_train = indexes[n_test : (n_test + n_train)]  # noqa: E203
            yield ind_train.tolist(), ind_test.tolist()


@overload
def train_test_split(
    *arrays: T,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    metric: Union[
        Metrics, Callable[[ArrayLike, ArrayLike], np.ndarray]
    ] = "euclidean",
    n_jobs: Optional[int] = None,
    device: Device = "cpu",
) -> list[T]: ...


def train_test_split(
    *arrays: T,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    metric: Union[
        Metrics, Callable[[ArrayLike, ArrayLike], np.ndarray]
    ] = "euclidean",
    n_jobs: Optional[int] = None,
    device: Device = "cpu",
    random_state: None = None,
    shuffle: None = None,
) -> list[T]:
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

    metric : Union[Metrics, Callable[[ArrayLike, ArrayLike], np.ndarray]]
        , optional

        The distance metric to use. See the documentation of
        - `scipy.spatial.distance.pdist`
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        - `sklearn.metrics.pairwise_distances`
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

        for valid values.
        , by default "euclidean"

        Valid values for metric are:

        - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1',
            'l2', 'manhattan']. These metrics support sparse matrix inputs.
            ['nan_euclidean'] but it does not yet support sparse matrices.
        - From scipy.spatial.distance: ['braycurtis', 'canberra',
            'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
            'mahalanobis', 'minkowski', 'rogerstanimoto',
            'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
            'sqeuclidean', 'yule'] See the documentation for
            scipy.spatial.distance for details on these metrics.
            These metrics do not support sparse matrix inputs.

        If you want to use GPU when calculating the distance matrix
        ('euclidean', 'manhattan', 'chebyshev' and 'minkowski'),
        you need to install 'pytorch' and set `device` to 'cuda' or 'mps'.

    n_jobs : int, optional
        The number of parallel jobs. It is valid only when CPU is used.
        , by default None

    device : Literal['cpu', 'cuda', 'mps'] or torch.device or str, optional
        , by default 'cpu'

        If you want to use GPU when calculating the distance matrix
        ('euclidean', 'manhattan', 'chebyshev' and 'minkowski'),
        you need to install 'pytorch' and set `device` to 'cuda' or 'mps'.

    random_state : None, deprecated
        This parameter is deprecated and has no effect
        because the algorithm is deterministic.

    shuffle : None, deprecated
        This parameter is deprecated and has no effect
        because the algorithm is deterministic.

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
            IgnoredArgumentWarning,
        )

    if random_state is not None:
        warnings.warn(
            "`random_state` is unnecessary since it is uniquely determined"
            " in this algorithm.",
            IgnoredArgumentWarning,
        )

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    cv = KSSplit(
        test_size=n_test,
        train_size=n_train,
        metric=metric,
        n_jobs=n_jobs,
        device=device,
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
        metric: Union[
            Metrics, Callable[[ArrayLike, ArrayLike], np.ndarray]
        ] = "euclidean",
        n_jobs: Optional[int] = None,
        device: Device = "cpu",
    ) -> None:
        """The root program of the Kennard-Stone algorithm,
        an algorithm for evenly partitioning data.

        Parameters
        ----------
        n_groups : int, optional
            how many groups to divide, by default 1

        scale : bool, optional
            scaling X or not, by default True

        metric : Union[Metrics, Callable[[ArrayLike, ArrayLike], np.ndarray]]
            , optional

            The distance metric to use. See the documentation of
            - `scipy.spatial.distance.pdist`
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
            - `sklearn.metrics.pairwise_distances`
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

            for valid values.
            , by default "euclidean"

            Valid values for metric are:

            - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1',
                'l2', 'manhattan']. These metrics support sparse matrix inputs.
                ['nan_euclidean'] but it does not yet support sparse matrices.
            - From scipy.spatial.distance: ['braycurtis', 'canberra',
                'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
                'mahalanobis', 'minkowski', 'rogerstanimoto',
                'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                'sqeuclidean', 'yule'] See the documentation for
                scipy.spatial.distance for details on these metrics.
                These metrics do not support sparse matrix inputs.

            If you want to use GPU when calculating the distance matrix
            ('euclidean', 'manhattan', 'chebyshev' and 'minkowski'),
            you need to install 'pytorch' and set `device` to 'cuda' or 'mps'.

        n_jobs : int, optional
            The number of parallel jobs. It is valid only when CPU is used.
            , by default None

        device : Literal['cpu', 'cuda', 'mps'] or torch.device or str, optional
            , by default 'cpu'

            If you want to use GPU when calculating the distance matrix
            ('euclidean', 'manhattan', 'chebyshev' and 'minkowski'),
            you need to install 'pytorch' and set `device` to 'cuda' or 'mps'.
        """
        self.n_groups = n_groups
        self.scale = scale
        self.metric = metric
        self.n_jobs = n_jobs
        self.device = device

    def get_indexes(self, X: ArrayLike) -> list[array[int]]:
        """Sort indexes by the Kennard-Stone algorithm.

        Parameters
        ----------
        X : ArrayLike
            The data to be sorted.

        Returns
        -------
        list[array[int]]
            The sorted indexes.
        """
        # check input array
        X_checked: np.ndarray = check_array(
            X,
            ensure_2d=True,
            dtype="numeric",
            force_all_finite="allow-nan"
            if self.metric == "nan_euclidean"
            else True,
        )
        n_samples = X_checked.shape[0]

        # drop no variance
        vselector = VarianceThreshold(threshold=0.0)
        X_checked = vselector.fit_transform(X_checked)

        if self.scale:
            scaler = StandardScaler()
            X_checked = scaler.fit_transform(X_checked)

        # Save the original X_checked.
        # self._original_X = X_checked.copy()

        # Pre-calculate the distance matrix.
        self.distance_matrix = pairwise_distances(
            X_checked,
            metric=self.metric,
            n_jobs=self.n_jobs,
            device=self.device,
        )

        # 全ての組成に対してそれぞれの平均との距離の二乗を配列として得る． (サンプル数の分だけ存在)
        # distance_to_ave = np.sum(np.square(X_checked - X_checked.mean(axis=0)), axis=1)
        kwargs_pairwise_distances: dict[str, Any] = dict()
        if self.metric == "mahalanobis":
            kwargs_pairwise_distances["VI"] = np.linalg.inv(
                np.cov(X_checked, rowvar=False)
            )
        elif self.metric == "seuclidean":
            kwargs_pairwise_distances["V"] = np.var(X_checked, axis=0, ddof=1)

        distance_to_ave = pairwise_distances(
            X_checked,
            X_checked.mean(axis=0, keepdims=True),
            metric=self.metric,
            n_jobs=self.n_jobs,
            device=self.device,
            **kwargs_pairwise_distances,
        ).ravel()

        # 最大値を取るサンプル (平均からの距離が一番遠い) のindex_numberを保存
        idx_farthest: array[int] = array(
            "i", np.argsort(distance_to_ave)[::-1][: self.n_groups].tolist()
        )

        distance_min = self.distance_matrix[idx_farthest, :]

        # params
        indexes_selected = idx_farthest
        lst_indexes_selected_prev: list[array[int]] = [
            array("L") for _ in range(self.n_groups)
        ]
        indexes_remaining_prev = array("L", range(n_samples))

        for _ in range(
            n_samples // self.n_groups + bool(n_samples % self.n_groups) - 1
        ):
            # collect the current indexes
            indexes_remaining: array[int] = array("L")
            arg_selected: array[int] = array("L")
            for j, idx in enumerate(indexes_remaining_prev):
                if idx in set(indexes_selected):
                    arg_selected.append(j)
                else:
                    indexes_remaining.append(idx)
            n_remaining = len(indexes_remaining)

            lst_indexes_selected = [
                indexes_selected_prev + array("L", (index_selected,))
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

            distance_min = np.min(
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

            _st_arg_delete: set[int] = set()
            indexes_selected_next: array[int] = array("L")
            for k in range(self.n_groups):
                if k == 0:
                    arg_delete = np.argmax(
                        distance_min[k],
                    ).item()
                elif 0 < n_remaining - k:
                    sorted_args = array(
                        "L",
                        np.argsort(
                            distance_min[k],
                        ).tolist(),
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
            indexes_output: list[array[int]] = []
            for k in range(self.n_groups):
                indexes_selected_reversed = lst_indexes_selected[k][::-1]
                if k < len(indexes_selected_next):
                    index_selected_next = indexes_selected_next[k]
                    indexes_output.append(
                        array("L", (index_selected_next,))
                        + indexes_selected_reversed
                    )
                else:
                    indexes_output.append(indexes_selected_reversed)
            return indexes_output


if __name__ == "__main__":
    pass
