"""
Copyright © 2021 yu9824
"""

from __future__ import annotations

import sys
import warnings
from array import array
from itertools import chain
from typing import Any, Optional, TypeVar, Union

if sys.version_info >= (3, 9):
    from collections.abc import Callable, Generator
else:
    from typing import Callable, Generator


from inspect import signature

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
        """Kennard–Stone based K-Fold cross-validator.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds. Must be at least 2.

        metric : {Metrics, callable}, default="euclidean"
            Distance metric. Either a metric name accepted by
            ``sklearn.metrics.pairwise_distances`` /
            ``scipy.spatial.distance.pdist`` or a callable returning an
            ``ndarray``. With GPU ('euclidean', 'manhattan', 'chebyshev',
            'minkowski'), ``torch.cdist`` is used.

        n_jobs : int, default=None
            Number of parallel jobs (CPU only).

        device : {"cpu", "cuda", "mps"} or torch.device or str, default="cpu"
            Device for distance matrix computation.

        random_state : None, deprecated
            No effect (algorithm is deterministic).

        shuffle : None, deprecated
            No effect (algorithm is deterministic).
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
        """Splitting helper for train/test based on Kennard–Stone.

        Parameters
        ----------
        n_splits : int, default=1
            Must be 1 for this class.

        test_size : float or int, default=None
            Same semantics as ``sklearn.model_selection.train_test_split``.

        train_size : float or int, default=None
            Same semantics as ``sklearn.model_selection.train_test_split``.

        metric, n_jobs, device : see also ``KFold``
        """
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
        """Internal generator. Yields train/test indices."""
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
    """Split arrays or matrices into train and test subsets via Kennard–Stone.

    The first input array determines the geometric order of indices so that the
    split is as uniform as possible. All subsequent arrays are split using the
    same indices.

    Parameters
    ----------
    *arrays : sequence of indexables
        Arrays of equal length (list, ndarray, scipy-sparse, pandas DataFrame, etc.).

    test_size : float or int, default=None
        Proportion in [0.0, 1.0] or absolute count. If ``None``, it becomes the
        complement of ``train_size``. If both are ``None``, defaults to 0.25.

    train_size : float or int, default=None
        Proportion or absolute count for the train split. If ``None``, becomes
        the complement of ``test_size``.

    metric, n_jobs, device : see also ``KFold``

    random_state, shuffle : None, deprecated
        No effect (algorithm is deterministic).

    Returns
    -------
    list
        A list like ``[X_train, X_test, y_train, y_test, ...]``.

    Raises
    ------
    ValueError
        If no input arrays are provided.
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
        """Internal class implementing the core of the Kennard–Stone algorithm.

        Parameters
        ----------
        n_groups : int, default=1
            Number of groups to split into.

        scale : bool, default=True
            Whether to standardize features before computing distances.

        metric, n_jobs, device : see also ``KFold``
        """
        self.n_groups = n_groups
        self.scale = scale
        self.metric = metric
        self.n_jobs = n_jobs
        self.device = device

    def get_indexes(self, X: ArrayLike) -> list[array[int]]:
        """Compute index sequences using the Kennard–Stone procedure.

        Parameters
        ----------
        X : ArrayLike
            2D array of shape (n_samples, n_features).

        Returns
        -------
        list[array[int]]
            A list of index arrays corresponding to each group.
        """
        # check input array
        # scikit-learn 1.6+ deprecates 'force_all_finite' and 1.8 renames to
        # 'ensure_all_finite'. Check the signature dynamically.
        check_array_sig = signature(check_array)
        supports_ensure_all_finite = (
            "ensure_all_finite" in check_array_sig.parameters
        )
        supports_force_all_finite = (
            "force_all_finite" in check_array_sig.parameters
        )

        check_kwargs: dict[str, Any] = dict(
            ensure_2d=True,
            dtype="numeric",
        )
        if supports_ensure_all_finite:
            check_kwargs["ensure_all_finite"] = (
                "allow-nan" if self.metric == "nan_euclidean" else True
            )
        elif supports_force_all_finite:
            check_kwargs["force_all_finite"] = (
                "allow-nan" if self.metric == "nan_euclidean" else True
            )

        try:
            X_checked: np.ndarray = check_array(
                X,
                **check_kwargs,
            )
        except TypeError:
            # Fallback when the argument is not accepted at runtime
            check_kwargs.pop("ensure_all_finite", None)
            check_kwargs.pop("force_all_finite", None)
            X_checked = check_array(
                X,
                **check_kwargs,
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
