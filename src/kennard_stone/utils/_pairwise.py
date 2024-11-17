import sys
import warnings
from typing import Optional, Union

if sys.version_info >= (3, 9):
    from collections.abc import Callable
else:
    from typing import Callable

import numpy as np
import sklearn.metrics.pairwise
from numpy.typing import ArrayLike

from ..logging import get_child_logger
from ._type_alias import Device, Metrics
from ._utils import is_installed

_logger = get_child_logger(__name__)


def pairwise_distances(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    metric: Union[
        Metrics, Callable[[ArrayLike, ArrayLike], np.ndarray]
    ] = "euclidean",
    n_jobs: Optional[int] = None,
    force_all_finite=True,
    device: Device = "cpu",
    verbose: int = 1,
    **kwargs,
) -> np.ndarray:
    """Wrapper function for 'sklearn.metrics.pairwise.pairwise_distances' and
    'torch.cdist'.

    This function is used to calculate pairwise distances
    between two sets of points.

    Parameters
    ----------
    X : array-like of shape (n_samples_1, n_features)
        Array of points.

    Y : array-like of shape (n_samples_2, n_features), default=None
        Array of points. If None, the distance between X and itself is
        calculated.

    metric : str or callable, default="euclidean"
        The metric to use when calculating distance between instances
        in a feature array.

        If metric is a string, it must be one of the options allowed by
        sklearn.metrics.pairwise.pairwise_distances.

        if you want to use PyTorch's distance function, you can use
        'manhattan', 'euclidean', 'chebyshev', 'minowski' as a metric.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel. (Note: 'n_jobs' is not supported by PyTorch.)

    force_all_finite : bool, default=True
        Whether to raise an error on np.inf and np.nan in X.

    device : Literal['cpu', 'cuda', 'mps'] or torch.device or str
    , default="cpu"
        Device to use for calculating pairwise distances.

    Returns
    -------
    distance_X : ndarray of shape (n_samples_1, n_samples_2)
        Array of distances.
    """
    if is_installed("torch"):
        import torch  # type: ignore

        device = torch.device(device)
        available_torch = device.type != "cpu"
    else:
        available_torch = False

    if available_torch:
        # Convert NumPy array to PyTorch tensor and move it to GPU
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        if Y is not None:
            Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)
        else:
            Y_tensor = X_tensor

        p: Union[int, float, None]
        # Calculate pairwise distances on GPU
        if metric == "manhattan":
            p = 1
        elif metric == "euclidean":
            p = 2
        elif metric == "chebyshev":
            p = float("inf")
        elif metric == "minowski":
            p = kwargs.get("p", 2)
        else:
            p = None
            warnings.warn(f"{metric} is not supported by PyTorch. ")
    else:
        p = None

    if p is None:
        if verbose > 0:
            _logger.info(
                "Calculating pairwise distances using scikit-learn.\n"
            )
        return sklearn.metrics.pairwise.pairwise_distances(
            X,
            Y=Y,
            metric=metric,
            n_jobs=n_jobs,
            force_all_finite=force_all_finite,
            **kwargs,
        )
    else:
        if verbose > 0:
            _logger.info(
                f"Calculating pairwise distances using PyTorch on '{device}'."
            )
        disntace_tensor = torch.cdist(X_tensor, Y_tensor, p=p)
        if torch.allclose(X_tensor, Y_tensor):
            disntace_tensor = disntace_tensor.fill_diagonal_(0)
        return disntace_tensor.cpu().numpy()
