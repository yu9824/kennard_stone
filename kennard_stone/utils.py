from typing import Optional, Union
import pkgutil
import sys
import warnings

import numpy as np
from numpy.typing import ArrayLike
import sklearn.metrics.pairwise


class IgnoredArgumentWarning(Warning):
    """Warning used to ignore an argument."""

    ...


def is_installed(name: str) -> bool:
    """Check if a package is installed.
    Parameters
    ----------
    name : str
        Name of the package.
    Returns
    -------
    is_installed : bool
        Whether the package is installed.
    """
    return pkgutil.find_loader(name) is not None


if is_installed("torch"):
    import torch


if sys.version_info >= (3, 8) or is_installed("typing_extensions"):
    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal

    METRICS = Literal[
        "cityblock",
        "cosine",
        "euclidean",
        "l1",
        "l2",
        "manhattan",
        "braycurtis",
        "canberra",
        "chebyshev",
        "correlation",
        "dice",
        "hamming",
        "jaccard",
        "mahalanobis",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "sqeuclidean",
        "yule",
    ]
else:
    METRICS = str

if sys.version_info >= (3, 8) or is_installed("typing_extensions"):
    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal

    DEVICE = (
        Union[torch.device, Literal["cpu", "cuda", "mps"], str]
        if is_installed("torch")
        else Literal["cpu"]
    )
else:
    DEVICE = Union[torch.device, str] if is_installed("torch") else str


def pairwise_distances(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    metric: METRICS = "euclidean",
    n_jobs: Optional[int] = None,
    force_all_finite=True,
    device: DEVICE = "cpu",
    verbose: int = 1,
    **kwds,
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
        import torch

        device = torch.device(device)
        available_torch = device.type != "cpu"
    else:
        available_torch = False

    if available_torch:
        # Convert NumPy array to PyTorch tensor and move it to GPU
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        if Y is not None:
            Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        else:
            Y_tensor = X_tensor

        # Calculate pairwise distances on GPU
        if metric == "manhattan":
            p = 1
        elif metric == "euclidean":
            p = 2
        elif metric == "chebyshev":
            p = float("inf")
        elif metric == "minowski":
            p = kwds.get("p", 2)
        else:
            p = None
            warnings.warn(f"{metric} is not supported by PyTorch. ")
    else:
        p = None

    if p is None:
        if verbose > 0:
            sys.stdout.write(
                "Calculating pairwise distances using scikit-learn.\n"
            )
        return sklearn.metrics.pairwise.pairwise_distances(
            X,
            Y=Y,
            metric=metric,
            n_jobs=n_jobs,
            force_all_finite=force_all_finite,
            **kwds,
        )
    else:
        if verbose > 0:
            sys.stdout.write(
                "Calculating pairwise distances using PyTorch"
                f" on '{device.type}'.\n"
            )
        disntace_tensor = torch.cdist(X_tensor, Y_tensor, p=p)
        if torch.allclose(X_tensor, Y_tensor):
            disntace_tensor = disntace_tensor.fill_diagonal_(0)
        return disntace_tensor.cpu().numpy()


if __name__ == "__main__":
    import numpy as np

    print(
        np.allclose(
            pairwise_distances([[1, 2], [3, 4]], device="mps"),
            pairwise_distances([[1, 2], [3, 4]], device="cpu"),
        )
    )
