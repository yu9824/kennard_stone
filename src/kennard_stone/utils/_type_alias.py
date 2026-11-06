"""Type aliases used across the package.

This module defines ``Device`` for GPU/CPU device specification,
``Metrics`` as a literal set of metric names that can be used for distance
computation, and ``MetricLike`` accepting either a metric name or a callable.

Notes
-----
Representative public types referenced by the API are centralized here for
clear Sphinx rendering.
"""

from __future__ import annotations

import sys

# NOTE: Type aliases below are evaluated at runtime, so they must keep using
# ``typing.Union`` / ``typing.Callable`` instead of ``X | Y`` /
# ``collections.abc.Callable[...]`` while Python 3.8 and 3.9 are supported.
# Switch to the PEP 604 / PEP 585 syntax after dropping Python 3.8 and 3.9.
from typing import Callable, Literal, Union

import numpy as np
from numpy.typing import ArrayLike

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from ._utils import is_installed

if is_installed("torch"):
    import torch  # type: ignore

    Device: TypeAlias = Union[
        "torch.device", str, Literal["cpu", "cuda", "mps"]
    ]
else:
    Device: TypeAlias = Union[str, Literal["cpu"]]  # type: ignore[misc,no-redef]

Metrics: TypeAlias = Literal[
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

MetricLike: TypeAlias = Union[
    Metrics, Callable[[ArrayLike, ArrayLike], np.ndarray]
]
"""Metric name or a callable returning a distance matrix."""
