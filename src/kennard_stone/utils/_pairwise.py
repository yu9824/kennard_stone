import sys
import warnings
from typing import Optional, Union

if sys.version_info >= (3, 9):
    from collections.abc import Callable
else:
    from typing import Callable

from inspect import signature

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
    ensure_all_finite: bool = True,
    force_all_finite: Optional[bool] = None,
    device: Device = "cpu",
    verbose: int = 1,
    **kwargs,
) -> np.ndarray:
    """Compute the pairwise distance matrix between two point sets.

    Wraps scikit-learn's ``pairwise_distances`` and PyTorch's ``torch.cdist``
    and chooses an implementation depending on the environment and device.

    Parameters
    ----------
    X : array-like of shape (n_samples_1, n_features)
        First set of points.

    Y : array-like of shape (n_samples_2, n_features), default=None
        Second set of points. If ``None``, distances within ``X`` are
        computed.

    metric : str or callable, default="euclidean"
        Distance metric. If a string, it must be accepted by scikit-learn.
        With PyTorch, you can use 'manhattan', 'euclidean', 'chebyshev',
        or 'minowski'.

    n_jobs : int, default=None
        Number of parallel jobs (effective with scikit-learn only).

    ensure_all_finite : bool, default=True
        Whether to raise on inf/NaN.

    force_all_finite : Optional[bool], default=None
        Deprecated alias of ``ensure_all_finite``. If provided, overrides it.

    device : {"cpu", "cuda", "mps"} or torch.device or str, default="cpu"
        Device to use for computation.

    Returns
    -------
    ndarray of shape (n_samples_1, n_samples_2)
        Distance matrix.
    """
    if is_installed("torch"):
        import torch  # type: ignore

        device = torch.device(device)
        available_torch = device.type != "cpu"
    else:
        available_torch = False

    # Handle deprecated alias
    if force_all_finite is not None:
        warnings.warn(
            "'force_all_finite' is deprecated. Use 'ensure_all_finite' instead.",
            DeprecationWarning,
        )
        ensure_all_finite = force_all_finite

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
        # scikit-learn 1.6+ deprecates 'force_all_finite' and 1.8 renames to
        # 'ensure_all_finite'. Dynamically use whichever is available.
        pd_sig = signature(sklearn.metrics.pairwise.pairwise_distances)
        supports_ensure_all_finite = "ensure_all_finite" in pd_sig.parameters
        supports_force_all_finite = "force_all_finite" in pd_sig.parameters

        call_kwargs = dict(
            Y=Y,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs,
        )
        if supports_ensure_all_finite:
            call_kwargs["ensure_all_finite"] = ensure_all_finite
        elif supports_force_all_finite:
            call_kwargs["force_all_finite"] = ensure_all_finite

        try:
            return sklearn.metrics.pairwise.pairwise_distances(
                X, **call_kwargs
            )
        except TypeError:
            # Fallback for environments where the arg is rejected at runtime
            call_kwargs.pop("ensure_all_finite", None)
            call_kwargs.pop("force_all_finite", None)
            return sklearn.metrics.pairwise.pairwise_distances(
                X, **call_kwargs
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
