"""
Copyright © 2021 yu9824
"""

from __future__ import annotations

import warnings
from array import array
from collections.abc import Generator
from itertools import chain
from typing import Any, TypeVar

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

from ..utils._compat import finite_check_kwargs
from ..utils._pairwise import pairwise_distances
from ..utils._type_alias import Device, InitMethod, MetricLike
from ..utils._utils import (
    IgnoredArgumentWarning,
)

# for typing
T = TypeVar("T")

INIT_METHODS = ("farthest_from_mean", "farthest_pair")

# TODO: Update docstrings


class KFold(_BaseKFold):
    def __init__(
        self,
        n_splits: int = 5,
        *,
        metric: MetricLike = "euclidean",
        n_jobs: int | None = None,
        device: Device = "cpu",
        scale: bool = True,
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

        scale : bool, default=True
            Whether to standardize features (``StandardScaler``) before
            computing distances. Features with zero variance are always
            dropped.

        random_state : None, deprecated
            No effect (algorithm is deterministic).

        shuffle : None, deprecated
            No effect (algorithm is deterministic).
        """
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.metric = metric
        self.n_jobs = n_jobs
        self.device = device
        self.scale = scale

        if shuffle is not None:
            warnings.warn(
                "`shuffle` is unnecessary because it is always shuffled"
                " in this algorithm.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )
        del self.shuffle

        if random_state is not None:
            warnings.warn(
                "`random_state` is unnecessary since it is uniquely determined"
                " in this algorithm.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )
        del self.random_state

    def _iter_test_indices(
        self, X=None, y=None, groups=None
    ) -> Generator[array[int], None, None]:
        ks = _KennardStone(
            n_groups=self.get_n_splits(),
            scale=self.scale,
            metric=self.metric,
            n_jobs=self.n_jobs,
            device=self.device,
        )
        yield from ks.get_indexes(X)


class KSSplit(BaseShuffleSplit):
    def __init__(
        self,
        n_splits: int = 1,
        *,
        test_size: float | int | None = None,  # noqa: PYI041
        train_size: float | int | None = None,  # noqa: PYI041
        metric: MetricLike = "euclidean",
        n_jobs: int | None = None,
        device: Device = "cpu",
        scale: bool = True,
        init: InitMethod = "farthest_from_mean",
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

        metric, n_jobs, device, scale : see also ``KFold``

        init : {"farthest_from_mean", "farthest_pair"}, default="farthest_from_mean"
            How to choose the first selected samples. Selection then proceeds
            greedily by the max-min distance criterion in both cases.

            - ``"farthest_from_mean"``: start from the sample farthest from the
              mean of the (preprocessed) data.
            - ``"farthest_pair"``: start from the two samples with the largest
              mutual distance, as in the original Kennard–Stone algorithm
              (Kennard & Stone, 1969).
        """
        super().__init__(
            n_splits=n_splits, test_size=test_size, train_size=train_size
        )
        self.metric = metric
        self.n_jobs = n_jobs
        self.device = device
        self.scale = scale
        self.init = init

        assert self.get_n_splits() == 1, "n_splits must be 1"
        self._default_test_size = 0.1

    # overwrap abstractmethod
    def _iter_indices(
        self, X, y=None, groups=None
    ) -> Generator[tuple[list[int], list[int]], None, None]:
        """Internal generator. Yields train/test indices."""
        ks = _KennardStone(
            n_groups=1,
            scale=self.scale,
            metric=self.metric,
            n_jobs=self.n_jobs,
            device=self.device,
            init=self.init,
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
            yield ind_train.tolist(), ind_test.tolist()


def train_test_split(
    *arrays: T,
    test_size: float | int | None = None,  # noqa: PYI041
    train_size: float | int | None = None,  # noqa: PYI041
    metric: MetricLike = "euclidean",
    n_jobs: int | None = None,
    device: Device = "cpu",
    scale: bool = True,
    init: InitMethod = "farthest_from_mean",
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

    metric, n_jobs, device, scale : see also ``KFold``

    init : {"farthest_from_mean", "farthest_pair"}, default="farthest_from_mean"
        How to choose the first selected samples. Selection then proceeds
        greedily by the max-min distance criterion in both cases.

        - ``"farthest_from_mean"``: start from the sample farthest from the
          mean of the (preprocessed) data.
        - ``"farthest_pair"``: start from the two samples with the largest
          mutual distance, as in the original Kennard–Stone algorithm
          (Kennard & Stone, 1969).

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
            stacklevel=2,
        )

    if random_state is not None:
        warnings.warn(
            "`random_state` is unnecessary since it is uniquely determined"
            " in this algorithm.",
            IgnoredArgumentWarning,
            stacklevel=2,
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
        scale=scale,
        init=init,
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
        metric: MetricLike = "euclidean",
        n_jobs: int | None = None,
        device: Device = "cpu",
        init: InitMethod = "farthest_from_mean",
    ) -> None:
        """Internal class implementing the core of the Kennard–Stone algorithm.

        Parameters
        ----------
        n_groups : int, default=1
            Number of groups to split into.

        scale : bool, default=True
            Whether to standardize features before computing distances.

        metric, n_jobs, device : see also ``KFold``

        init : {"farthest_from_mean", "farthest_pair"}, default="farthest_from_mean"
            See ``train_test_split``. ``"farthest_pair"`` requires
            ``n_groups=1``.
        """
        self.n_groups = n_groups
        self.scale = scale
        self.metric = metric
        self.n_jobs = n_jobs
        self.device = device
        self.init = init

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
        if self.init not in INIT_METHODS:
            raise ValueError(
                f"init must be one of {INIT_METHODS}, got {self.init!r}."
            )
        if self.init == "farthest_pair" and self.n_groups != 1:
            raise ValueError(
                "init='farthest_pair' is only supported when n_groups=1, "
                f"got n_groups={self.n_groups}."
            )

        X_checked = _validate(X, self.metric)
        n_samples = X_checked.shape[0]
        if self.n_groups > n_samples:
            raise ValueError(
                f"n_groups={self.n_groups} must not exceed "
                f"n_samples={n_samples}."
            )

        X_processed = _preprocess(X_checked, self.scale)

        # Pre-calculate the distance matrix.
        self.distance_matrix = pairwise_distances(
            X_processed,
            metric=self.metric,
            n_jobs=self.n_jobs,
            device=self.device,
        )

        if self.init == "farthest_pair":
            initial_indexes = _farthest_pair_start(self.distance_matrix)
        else:
            initial_indexes = _farthest_from_mean(
                X_processed,
                self.n_groups,
                metric=self.metric,
                n_jobs=self.n_jobs,
                device=self.device,
            )
        return _greedy_select(
            self.distance_matrix, initial_indexes, self.n_groups
        )


def _validate(X: ArrayLike, metric: MetricLike) -> np.ndarray:
    """Validate ``X`` as a 2D numeric array (NaN allowed for nan_euclidean)."""
    return check_array(
        X,
        ensure_2d=True,
        dtype="numeric",
        **finite_check_kwargs(
            check_array,
            "allow-nan" if metric == "nan_euclidean" else True,
        ),
    )


def _preprocess(X: np.ndarray, scale: bool) -> np.ndarray:
    """Drop zero-variance features and optionally standardize."""
    X_processed = VarianceThreshold(threshold=0.0).fit_transform(X)
    if scale:
        X_processed = StandardScaler().fit_transform(X_processed)
    return X_processed


def _metric_kwargs(X: np.ndarray, metric: MetricLike) -> dict[str, Any]:
    """Extra parameters so that distances to a single point (the mean) use
    the statistics of the whole ``X``."""
    if metric == "mahalanobis":
        return {"VI": np.linalg.inv(np.cov(X, rowvar=False))}
    if metric == "seuclidean":
        return {"V": np.var(X, axis=0, ddof=1)}
    return {}


def _farthest_from_mean(
    X: np.ndarray,
    n_groups: int,
    *,
    metric: MetricLike,
    n_jobs: int | None,
    device: Device,
) -> array[int]:
    """Indexes of the ``n_groups`` samples farthest from the mean of ``X``,
    in descending order of distance."""
    distance_to_ave = pairwise_distances(
        X,
        X.mean(axis=0, keepdims=True),
        metric=metric,
        n_jobs=n_jobs,
        device=device,
        **_metric_kwargs(X, metric),
    ).ravel()
    return array("i", np.argsort(distance_to_ave)[::-1][:n_groups].tolist())


def _farthest_pair_start(distance_matrix: np.ndarray) -> array[int]:
    """Start of the original Kennard–Stone algorithm (single group).

    Returns ``i`` of the most distant pair ``(i, j)`` (the first maximum in
    row-major order). Greedy selection from ``i`` picks ``j`` next, because
    ``distance_matrix[i, j]`` is the global maximum and ``j`` is the first
    maximum in row ``i``. Hence the first two selected samples are the most
    distant pair.
    """
    i, _ = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    return array("i", (int(i),))


def _greedy_select(
    distance_matrix: np.ndarray,
    initial_indexes: array[int],
    n_groups: int,
) -> list[array[int]]:
    """Greedily assign samples to groups by the max-min distance criterion.

    Starting from ``initial_indexes`` (one per group), each group in turn
    takes the remaining sample whose distance to its nearest already-selected
    sample in that group is the largest.

    Returns
    -------
    list[array[int]]
        For each group, indexes in reverse order of selection.
    """
    n_samples = distance_matrix.shape[0]
    distance_min = distance_matrix[initial_indexes, :]

    # params
    indexes_selected = initial_indexes
    lst_indexes_selected_prev: list[array[int]] = [
        array("L") for _ in range(n_groups)
    ]
    indexes_remaining_prev = array("L", range(n_samples))

    # Values used after the loop when it runs zero times
    # (n_samples == n_groups): each group holds only its initial sample.
    lst_indexes_selected = lst_indexes_selected_prev
    indexes_selected_next = indexes_selected
    n_remaining = n_samples - len(indexes_selected)

    for _ in range(n_samples // n_groups + bool(n_samples % n_groups) - 1):
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
                lst_indexes_selected_prev, indexes_selected, strict=True
            )
        ]
        # /collect the current indexes

        # 代表長さを決定する
        distance_selected: np.ndarray = distance_matrix[
            np.ix_(indexes_selected, indexes_remaining)
        ]
        distance_min = np.delete(distance_min, arg_selected, axis=1)

        distance_min = np.min(
            np.concatenate(
                [
                    distance_selected.reshape(n_groups, 1, -1),
                    distance_min.reshape(n_groups, 1, -1),
                ],
                axis=1,
            ),
            axis=1,
        )

        # まだ選択されていない各サンプルにおいて、これまで選択されたすべてのサンプルとの間で
        # ユークリッド距離を計算し，その最小の値を「代表長さ」とする．

        _st_arg_delete: set[int] = set()
        indexes_selected_next: array[int] = array("L")
        for k in range(n_groups):
            if k == 0:
                arg_delete = np.argmax(
                    distance_min[k],
                ).item()
            elif n_remaining - k > 0:
                sorted_args = array(
                    "L",
                    np.argsort(
                        distance_min[k],
                    ).tolist(),
                )
                # Take the sample with the largest representative distance
                # among those not taken by earlier groups in this round.
                # At most k samples are taken and n_remaining > k, so a free
                # sample always exists.
                for j in range(n_remaining - 1, -1, -1):
                    arg_delete = sorted_args[j]
                    if arg_delete not in _st_arg_delete:
                        break
                else:
                    raise RuntimeError(
                        f"No free sample for group {k} although "
                        f"{n_remaining} samples remain. This is a bug."
                    )
            else:
                break

            _st_arg_delete.add(arg_delete)
            index_selected: int = indexes_remaining[arg_delete]

            indexes_selected_next.append(index_selected)

        indexes_selected = indexes_selected_next
        lst_indexes_selected_prev = lst_indexes_selected
        indexes_remaining_prev = indexes_remaining
    # もうないなら遠い順から近い順 (test側) に並べ替えて終える
    assert n_remaining - len(indexes_selected_next) <= 0
    indexes_output: list[array[int]] = []
    for k in range(n_groups):
        indexes_selected_reversed = lst_indexes_selected[k][::-1]
        if k < len(indexes_selected_next):
            index_selected_next = indexes_selected_next[k]
            indexes_output.append(
                array("L", (index_selected_next,)) + indexes_selected_reversed
            )
        else:
            indexes_output.append(indexes_selected_reversed)
    return indexes_output


if __name__ == "__main__":
    pass
