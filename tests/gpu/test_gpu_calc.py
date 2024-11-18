import numpy as np
import pytest

from kennard_stone.utils import pairwise_distances
from kennard_stone.utils._utils import is_available_gpu


@pytest.mark.parametrize("metric", ("manhattan", "euclidean", "chebyshev"))
@pytest.mark.skipif(not is_available_gpu(), reason="GPU is not available.")
def test_gpu_manhattan(metric, prepare_data, get_device):
    X, _ = prepare_data
    X_mean = X.mean(axis=0, keepdims=True)

    # 通常の距離行列
    assert np.allclose(
        pairwise_distances(X, metric=metric),
        pairwise_distances(X, metric=metric, device=get_device),
    )

    # 平均との行列
    assert np.allclose(
        pairwise_distances(X, X_mean, metric=metric),
        pairwise_distances(X, X_mean, metric=metric, device=get_device),
    )
