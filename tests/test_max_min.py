"""Each group must take its best free sample by the max-min criterion.

In each round, groups pick one sample in turn. A group must take, among the
samples not selected yet (including those taken by earlier groups in the
same round), the one whose distance to its nearest sample in the group is
the largest.
"""

from __future__ import annotations

import numpy as np
import pytest
from _regression_cases import CORE_CASES, DATASETS

from kennard_stone._core._core import _KennardStone


def _check_max_min(
    distance_matrix: np.ndarray, groups: list[list[int]]
) -> None:
    n_samples = distance_matrix.shape[0]
    # Each group is returned in reverse order of selection.
    orders = [group[::-1] for group in groups]
    selected = [[order[0]] for order in orders]
    taken = {order[0] for order in orders}

    for r in range(1, max(len(order) for order in orders)):
        for k, order in enumerate(orders):
            if r >= len(order):
                continue
            free = np.array([i for i in range(n_samples) if i not in taken])
            distance_min = distance_matrix[np.ix_(selected[k], free)].min(
                axis=0
            )
            picked = order[r]
            assert picked in free, f"round {r}, group {k}: {picked} was taken"
            best = distance_min.max()
            got = distance_min[free.tolist().index(picked)]
            assert got >= best - 1e-9 * max(best, 1.0), (
                f"round {r}, group {k}: picked {picked} at {got}, "
                f"best free is {free[distance_min.argmax()]} at {best}"
            )
            selected[k].append(picked)
            taken.add(picked)


def _run(X: np.ndarray, **kwargs) -> tuple[np.ndarray, list[list[int]]]:
    ks = _KennardStone(**kwargs)
    groups = [[int(i) for i in g] for g in ks.get_indexes(X)]
    return ks.distance_matrix, groups


@pytest.mark.parametrize(
    ("name", "dataset", "kwargs"),
    CORE_CASES,
    ids=[name for name, _, _ in CORE_CASES],
)
def test_core_cases_follow_max_min(name, dataset, kwargs):
    _check_max_min(*_run(DATASETS[dataset](), **kwargs))


@pytest.mark.parametrize("n_groups", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(20))
def test_random_data_follow_max_min(seed, n_groups):
    rng = np.random.RandomState(seed)
    n_samples = rng.randint(6, 60)
    if seed % 2:
        # Integer data with many tied distances.
        X = rng.randint(0, 5, size=(n_samples, 2)).astype(float)
    else:
        X = rng.rand(n_samples, 3)

    _check_max_min(*_run(X, n_groups=n_groups))
