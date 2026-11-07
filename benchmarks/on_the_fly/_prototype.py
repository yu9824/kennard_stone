"""Prototype: Kennard–Stone selection without the full distance matrix.

Issue #41. ``greedy_select_on_the_fly`` is a verbatim copy of
``kennard_stone._core._core._greedy_select`` except for the two places
marked CHANGED, where distances are computed on the fly instead of read
from a precomputed matrix. Everything else (including tie-breaking) is
unchanged, so the result is identical whenever the distances are.

Not part of the package; used by check_equivalence.py and bench.py.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable

import numpy as np
from joblib import effective_n_jobs
from threadpoolctl import threadpool_limits

from kennard_stone._core._core import (
    INIT_METHODS,
    _farthest_from_mean,
    _metric_kwargs,
    _preprocess,
    _validate,
)
from kennard_stone.utils._pairwise import pairwise_distances


def greedy_select_on_the_fly(
    distance_rows: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_samples: int,
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
    # CHANGED: distances are computed on the fly instead of read from a
    # precomputed matrix. distance_rows(rows, cols) returns the distances
    # between the samples ``rows`` and ``cols`` (index arrays).
    distance_min = distance_rows(
        np.asarray(initial_indexes), np.arange(n_samples)
    )

    # params
    indexes_selected = initial_indexes
    lst_indexes_selected_prev: list[array[int]] = [
        array("L") for _ in range(n_groups)
    ]
    indexes_remaining_prev = array("L", range(n_samples))

    # Values used after the loop when it runs zero times
    # (n_samples == n_groups): each group holds only its initial sample.
    lst_indexes_selected = lst_indexes_selected_prev
    indexes_selected_next: array[int] = indexes_selected
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
        # CHANGED: see above.
        distance_selected: np.ndarray = distance_rows(
            np.asarray(indexes_selected), np.asarray(indexes_remaining)
        )
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
        indexes_selected_next = array("L")
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


def _cpu_distance_rows(X, metric, kwargs):
    def distance_rows(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return pairwise_distances(
            X[rows], X[cols], metric=metric, verbose=0, **kwargs
        )

    return distance_rows


def _gpu_distance_rows(X, metric, device):
    # Keep X on the device once instead of copying it at every step.
    import torch

    p = {"manhattan": 1, "euclidean": 2, "chebyshev": float("inf")}[metric]
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    def distance_rows(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        # Index arrays come from array("L") (unsigned), which torch does
        # not accept as indices.
        rows_t = torch.as_tensor(rows.astype(np.int64), device=device)
        cols_t = torch.as_tensor(cols.astype(np.int64), device=device)
        return (
            torch.cdist(X_tensor[rows_t], X_tensor[cols_t], p=p).cpu().numpy()
        )

    return distance_rows


def _farthest_pair_blockwise(distance_rows, n_samples, block=1024):
    """Find the start of ``init="farthest_pair"`` block by block.

    The result equals ``_farthest_pair_start`` on the full matrix.

    np.argmax on the full matrix returns the first maximum in row-major
    order; scanning row blocks in order and keeping the first strict
    maximum gives the same row.
    """
    best_value, best_row = -np.inf, 0
    cols = np.arange(n_samples)
    for start in range(0, n_samples, block):
        rows = np.arange(start, min(start + block, n_samples))
        d = distance_rows(rows, cols)
        flat = int(np.argmax(d))
        value = d.flat[flat]
        if value > best_value:
            best_value, best_row = value, start + flat // n_samples
    return array("L", (best_row,))


def kennard_stone_on_the_fly(
    X,
    n_groups: int = 1,
    scale: bool = True,
    metric: str = "euclidean",
    n_jobs: int | None = None,
    device: str = "cpu",
    init: str = "farthest_from_mean",
) -> list[array[int]]:
    """Counterpart of ``_KennardStone(...).get_indexes(X)``.

    ``n_jobs`` is the number of CPUs to use (None means 1, -1 all, as in
    scikit-learn). The distances are computed a few rows at a time, where
    joblib's overhead would dominate, so ``n_jobs`` limits the BLAS threads
    used by numpy instead.
    """
    with threadpool_limits(limits=effective_n_jobs(n_jobs)):
        return _kennard_stone_on_the_fly(
            X, n_groups, scale=scale, metric=metric, device=device, init=init
        )


def _kennard_stone_on_the_fly(
    X, n_groups, *, scale, metric, device, init
) -> list[array[int]]:
    if init not in INIT_METHODS:
        raise ValueError(f"init must be one of {INIT_METHODS}, got {init!r}.")
    X_checked = _validate(X, metric)
    n_samples = X_checked.shape[0]
    if n_groups > n_samples:
        raise ValueError("n_groups must not exceed n_samples.")
    X_processed = _preprocess(X_checked, scale)

    if device == "cpu":
        # The full-matrix path lets scipy/sklearn estimate the statistics
        # of mahalanobis/seuclidean from all samples; pass them explicitly
        # because only a subset of rows is given here.
        distance_rows = _cpu_distance_rows(
            X_processed, metric, _metric_kwargs(X_processed, metric)
        )
    else:
        distance_rows = _gpu_distance_rows(X_processed, metric, device)

    if init == "farthest_pair":
        initial_indexes = _farthest_pair_blockwise(distance_rows, n_samples)
    else:
        initial_indexes = _farthest_from_mean(
            X_processed, n_groups, metric=metric, n_jobs=None, device=device
        )
    return greedy_select_on_the_fly(
        distance_rows, n_samples, initial_indexes, n_groups
    )
