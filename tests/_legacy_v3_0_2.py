"""Frozen copy of ``_KennardStone`` from kennard_stone v3.0.2.

Used as the reference implementation in regression tests: the current
implementation must return exactly the same indexes in the same environment.
Comparing within the same environment keeps the test independent of how
numpy / scikit-learn versions break (near-)ties.

Only ``pairwise_distances`` is replaced by a CPU-only equivalent of the
scikit-learn path used in v3.0.2. Do not modify the class body.
"""

# flake8: noqa
# fmt: off
from __future__ import annotations

from array import array
from inspect import signature
from typing import Any, Callable, Optional, Union

import numpy as np
import sklearn.metrics.pairwise
from numpy.typing import ArrayLike
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

Metrics = str
Device = str


def pairwise_distances(X, Y=None, metric="euclidean", n_jobs=None, device="cpu", **kwargs):
    params = signature(sklearn.metrics.pairwise.pairwise_distances).parameters
    if "ensure_all_finite" in params:
        kwargs["ensure_all_finite"] = True
    elif "force_all_finite" in params:
        kwargs["force_all_finite"] = True
    return sklearn.metrics.pairwise.pairwise_distances(
        X, Y=Y, metric=metric, n_jobs=n_jobs, **kwargs
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


